
import sys
import time
from datetime import timedelta
from timeit import time

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyarrow import Table

import numpy as np
import numpy.ma as ma
import pandas as pd
from tqdm.contrib import tenumerate
from tqdm import tqdm
from scipy import sparse




from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors


import datamol as dm
import operator







dm.disable_rdkit_log()






def stopwatch(method):
    def timed(*args, **kw):
        ts = time.perf_counter()
        result = method(*args, **kw)
        te = time.perf_counter()
        duration = timedelta(seconds=te - ts)
        print(f"{method.__name__}: {duration}")
        return result
    return timed

def _preprocess(i, row):

    mol = dm.to_mol(str(row[smiles_column]), ordered=True)
    mol = dm.fix_mol(mol)
    mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
    mol = dm.standardize_mol(mol, disconnect_metals=False, normalize=True, reionize=True, uncharge=False, stereo=True)


    fingerprint_function = rdMolDescriptors.GetMorganFingerprintAsBitVect
    pars = { "radius": 2,
                     "nBits": 8192,
                     "invariants": [],
                     "fromAtoms": [],
                     "useChirality": False,
                     "useBondTypes": True,
                     "useFeatures": False,
            }
    fp = fingerprint_function(mol, **pars)

    row["standard_smiles"] = dm.standardize_smiles(dm.to_smiles(mol))
    row["selfies"] = dm.to_selfies(mol)
    row["inchi"] = dm.to_inchi(mol)
    row["inchikey"] = dm.to_inchikey(mol)
    row["onbits_fp"] =list(fp.GetOnBits())
    
    return row



@stopwatch
def fingerprint_matrix_from_df(df):
    smiles = list(df['enumerated_smiles'])
    onbits_fp = list(df['achiral_fp'])
    zincid = list(df['canonical_ID'])
    count_ligs = len(smiles)




    name_list =[]

    row_idx = list()
    col_idx = list()
    num_on_bits = []
    for count,m in enumerate(smiles):
        zincid_name = str(zincid[count])
        onbits = list(onbits_fp[count])

        col_idx+=onbits
        row_idx += [count]*len(onbits)
        num_bits = len(onbits)
        num_on_bits.append(num_bits)

        name_list.append(zincid_name)


    unfolded_size = 8192        
    fingerprint_matrix = sparse.coo_matrix((np.ones(len(row_idx)).astype(bool), (row_idx, col_idx)), 
              shape=(max(row_idx)+1, unfolded_size))
    fingerprint_matrix =  sparse.csr_matrix(fingerprint_matrix)
    fp_mat = fingerprint_matrix
    return fp_mat



def fast_jaccard(X, Y=None):
    """credit: https://stackoverflow.com/questions/32805916/compute-jaccard-distances-on-sparse-matrix"""
    if isinstance(X, np.ndarray):
        X = sparse.csr_matrix(X)
    if Y is None:
        Y = X
    else:
        if isinstance(Y, np.ndarray):
            Y = sparse.csr_matrix(Y)
    assert X.shape[1] == Y.shape[1]

    X = X.astype(bool).astype(int)
    Y = Y.astype(bool).astype(int)
    intersect = X.dot(Y.T)
    x_sum = X.sum(axis=1).A1
    y_sum = Y.sum(axis=1).A1
    xx, yy = np.meshgrid(x_sum, y_sum)
    union = ((xx + yy).T - intersect)
    return (1 - intersect / union).A





    
    
def min_distance(row):

    a = out_array[rowIndex(row)]
    minval = np.min(ma.masked_where(a==0, a))
    minvalpos = np.argmin(ma.masked_where(a==0, a))
#     sumval = np.sum(ma.masked_where(a==0, a))
    
    smiles_nn = smiles[minvalpos]
    name_nn = name[minvalpos]

    return smiles_nn, name_nn, minval


def rowIndex(row):
    return row.name


def table_to_csr_fp(table):    
    x = table.column('achiral_fp')


    row_idx = list()
    col_idx = list()
    num_on_bits = []

    for _ in range(len(x)):


        onbits = pa.ListValue.as_py(x[_])
        col_idx+=onbits
        row_idx += [_]*len(onbits)
        num_bits = len(onbits)
        num_on_bits.append(num_bits)




            # except:
            #     print('molecule failed')

    unfolded_size = 8192        
    fingerprint_matrix = sparse.coo_matrix((np.ones(len(row_idx)).astype(bool), (row_idx, col_idx)), shape=(max(row_idx)+1, unfolded_size))
    fingerprint_matrix =  sparse.csr_matrix(fingerprint_matrix)
    return fingerprint_matrix
  

@stopwatch
def i_want_to_go_faster(table):
    x = table.column('achiral_fp')
#     poop = pa.ChunkedArray.iterchunks(x)
    l = x
    col_idx = l.flatten().to_numpy()
    
    row_idx = l.value_parent_indices().to_numpy()

    
    unfolded_size = 8192        
    fingerprint_matrix = sparse.coo_matrix((np.ones(len(row_idx)).astype(bool), (row_idx, col_idx)), 
              shape=(max(row_idx)+1, unfolded_size))
    fingerprint_matrix =  sparse.csr_matrix(fingerprint_matrix)
    return fingerprint_matrix

data = [['OC1=CC=C(N=C(/C=C/C=C/C2=CN=C(NC)C=C2)S3)C3=C1', 'PBB3'], \
        ['OC(CF)COC1=CC=C(N=C(/C=C/C=C/C2=CN=C(NC)C=C2)S3)C3=C1', 'PM_PBB3'], \
        ['OC(CF)COC1=CC=C(N=C(/C=C/C#CC2=CN=C(NC)C=C2)S3)C3=C1', 'C5_05'],
       ]


# Create the pandas DataFrame
df = pd.DataFrame(data, columns = ['smiles', 'canonical_ID'])

smiles_column = 'smiles'

# run initial mapper on smiles column to generate basic information and fingerprint on bits
df_clean_mapped = dm.parallelized(_preprocess, list(df.iterrows()), arg_type='args', progress=True)
df_clean_mapped = pd.DataFrame(df_clean_mapped)
df_clean_mapped['enumerated_smiles'] = df_clean_mapped['standard_smiles']
df_clean_mapped['achiral_fp'] = df_clean_mapped['onbits_fp']
fingerprint_matrix_pbb3 = fingerprint_matrix_from_df(df_clean_mapped)

dataset = ds.dataset('batch0001', format="parquet")
fragments = [file for file in dataset.get_fragments()]
total_frags = len(fragments)
for count,record_batch in tenumerate(dataset.to_batches(columns=["canonical_ID", "enumerated_smiles", "achiral_fp"]), start=0, total=total_frags):
    
        #generate a CSR matrix of fingerprints for n rows in incoming record batch
        fingerprint_matrix_in = i_want_to_go_faster(record_batch)
        
        #
        y = record_batch.column('canonical_ID')
        name = [pa.StringScalar.as_py(y[_]) for _ in range(len(y))]
        
        
        z = record_batch.column('enumerated_smiles')
        smiles = [pa.StringScalar.as_py(z[_]) for _ in range(len(z))]
        
        
        out_array = fast_jaccard(fingerprint_matrix_in, fingerprint_matrix_pbb3)
        df_clean_mapped[['nn_smiles', 'nn_name', 'nn_distance']] = df_clean_mapped.apply(min_distance, axis=1, result_type='expand')
        
        
        cols_to_keep = ['canonical_ID', 'nn_smiles', 'nn_name', 'nn_distance', 'enumerated_smiles']
        df_clean_mapped = df_clean_mapped[cols_to_keep]
        table = pa.Table.from_pandas(df_clean_mapped, preserve_index=False)
        pq.write_table(table, f'/data/test_dataset/parquet_dataset/outfiles_faster/comparison_{count}.parquet')
