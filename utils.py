import pyarrow as pa
from pyarrow import csv
import pandas as pd
import pyarrow.feather as feather
from rdkit.Chem import PandasTools
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import numpy as np
import numpy.ma as ma
import tqdm
import pyarrow.parquet as pq
import numpy as np

import pandas as pd
import math
import time
from functools import reduce
import pyarrow.dataset as ds
from pyarrow import Table
from pyarrow import csv
import pyarrow as pa
from pyarrow.parquet import ParquetWriter
import pathlib

import datamol as dm
import operator
import sys
from datetime import timedelta
from timeit import time

from rich import print
import pathlib
from rich.console import Console


console = Console()

dm.disable_rdkit_log()



def _preprocess(i, row):
    '''Takes a smiles string and generates a clean rdkit mol with datamol. The stereoisomers
    are then enumerated while holding defined stereochemistry. Morgan fingerprints are then
    generated using RDkit with and without stereochemistry. The try/except logic deals with 
    RDkit mol failures on conversion of an invalid smiles string. Smarts are added for later
    searching.'''
    try:
        mol = dm.to_mol(str(row[smiles_column]), ordered=True)
        mol = dm.fix_mol(mol)
        mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
        mol = dm.standardize_mol(mol, disconnect_metals=False, normalize=True, reionize=True, uncharge=False, stereo=True)
        opts = StereoEnumerationOptions(unique=True,maxIsomers=20,rand=0xf00d)
        isomers = EnumerateStereoisomers(mol, options=opts)
        enum_smiles = sorted(Chem.MolToSmiles(y,isomericSmiles=True) for y in isomers)
#         enum_dm_smiles = sorted(dm.standardize_smiles(dm.to_smiles(x)) for x in isomers)
        
        smiles_list = []
        achiral_fp_lis = []
        chiral_fp_lis = []
        
#         standard_smiles_list = []
        for count, smi in enumerate(enum_smiles):
            smiles_string = smi
            
            mol = dm.to_mol(smi, ordered=True)
            mol = dm.fix_mol(mol)
            mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
            mol = dm.standardize_mol(mol, disconnect_metals=False, normalize=True, reionize=True, uncharge=False, stereo=True)

            fingerprint_function = rdMolDescriptors.GetMorganFingerprintAsBitVect
            
            pars = { "radius": 2,
                             "nBits": 8192,
                             "invariants": [],
                             "fromAtoms": [],
                             "useChirality": True,
                             "useBondTypes": True,
                             "useFeatures": False, }
            
            pars2 = { "radius": 2,
                             "nBits": 8192,
                             "invariants": [],
                             "fromAtoms": [],
                             "useChirality": False,
                             "useBondTypes": True,
                             "useFeatures": False, }

            fp = fingerprint_function(mol, **pars)
            fp1 = fingerprint_function(mol, **pars2)
            smiles_list.append(dm.standardize_smiles(smiles_string))
            achiral_fp_lis.append(list(fp1.GetOnBits()))
            chiral_fp_lis.append(list(fp.GetOnBits()))

        row["standard_smiles"] = dm.standardize_smiles(dm.to_smiles(mol))
        row["smarts"] = dm.to_smarts(mol)
        row["selfies"] = dm.to_selfies(mol)
        row["enumerated_smiles"] = smiles_list
        row["achiral_fp"] = achiral_fp_lis
        row["chiral_fp"] = chiral_fp_lis
#         row["dm_enumerated_smiles"] = enum_dm_smiles_lis
        # row["onbits_fp"] =list(fp.GetOnBits())
        
        return row

    except ValueError:
#         row["standard_smiles"] = 'dropped'
#         row["selfies"] = 'dropped'
#         row["inchi"] = 'dropped'
#         row["inchikey"] = 'dropped'
        
        row["standard_smiles"] = 'dropped'
        row["smarts"] = 'dropped'
        row["selfies"] = 'dropped'
        row["enumerated_smiles"] = list('dropped')
        row["achiral_fp"] = list('dropped')
        row["chiral_fp"] = list('dropped')
#         row["dm_enumerated_smiles"] = 'dropped'
        return row

def prep_parquet_db(df, n_jobs, smiles_col, catalog_id_col, canonical_id_col):
    
    '''Take a cleaned df that contains protonated/tautomerized smiles, 
    the vendor database ID and a canonical ID -- number indicates protomer/taut
    and 1) enumerate stereoisomers 2) generate chiral/achiral fingerprints 3) smarts and
    a new canonical ID that references stereoisomer. 
    
    Returns: elaborated dataframe - pandas dataframe
    
    args: df == dataframe to be passed in - pandas dataframe
    n_jobs == number of jobs utilized by joblib - integer
    smiles_col == the name of the smiles column - string
    catalog_id == name of column referencing the catalog ID - string
    canonical_id == name of col referencing the canonical ID usually Z123456789_1 where _1 is protomer/taut num -string
    '''
    
    smiles_column = smiles_col
    
    #Add clean the mols, standardize and generate lists for enumerated smiles, fingerprints both chiral/achiral at 8kbits
    df_clean_mapped = dm.parallelized(_preprocess, list(df.iterrows()), arg_type='args', progress=True, n_jobs=n_jobs)
    df_clean_mapped = pd.DataFrame(df_clean_mapped)
    
    #keep only the following columns
    columns_to_keep = ['enumerated_smiles', catalog_id_col, canonical_id_col, 'achiral_fp', 'chiral_fp', 'smarts', 'selfies']
    df2 = df_clean_mapped[columns_to_keep]
    
    #remove dropped smiles, these fail due to invalid mols from rdkit
    df_dropped = df2[df2.smarts == 'dropped']
    df3 = df2[df2.smarts != 'dropped']
    
    #explode all the lists and generate new rows, then drop duplicated smiles.
    df4 = df3.set_index(['CatalogID', 'ID_Index', 'smarts', 'selfies']).apply(pd.Series.explode).reset_index()
    df5 = df4.drop_duplicates(subset='enumerated_smiles', keep="first", inplace=False)
    df5 = df5.reset_index(drop=True)
    
    #generate a new indexing system that creates unique names for canonical_id Z123456_1_1 where 
    # Z123456_1 is taut/prot id and the additional _1 is the stereoisomer id
    df6 = df5.set_index('ID_Index')
    df6.index = df6.index + '_' + df6.groupby(level=0).cumcount().add(1).astype(str).replace('0','')
    df7 = df6.reset_index()
    
    #cleanup columns and return
    df7.columns = ['canonical_ID', 'CatalogID', 'smarts', 'selfies', 'enumerated_smiles', 'achiral_fp', 'chiral_fp']
    return df7
