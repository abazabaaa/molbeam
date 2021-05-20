

import os
import copy
import sys
import pathlib
from io import StringIO
import math
import time
from functools import reduce
from datetime import timedelta
from timeit import time

import pandas as pd
import pyarrow.feather as feather
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from datetime import timedelta
from timeit import time
import math
from rich import print
import pathlib
from rich.console import Console
import subprocess


import pyarrow.dataset as ds
from pyarrow import Table
from pyarrow import csv
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.parquet import ParquetWriter

import numpy as np
import numpy.ma as ma

from scipy import sparse

from rdkit.Chem import PandasTools
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from openbabel import pybel
from mdtraj.utils.delay_import import import_
import mols2grid

import re
import xml.etree.ElementTree as ET

import tqdm

import datamol as dm
import operator


dm.disable_rdkit_log()
#autocomplete wasn't working for some reason. This fixes it. 
%config Completer.use_jedi = False
console = Console()
os.environ["OE_LICENSE"] = "/data/openeye/oe_license.txt"

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
#     print('hello')
    mol = dm.to_mol(str(row[smiles_column]), ordered=True)
    mol = dm.fix_mol(mol)
    mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
    mol = dm.standardize_mol(mol, disconnect_metals=False, normalize=True, reionize=True, uncharge=False, stereo=True)

    fingerprint_function = rdMolDescriptors.GetMorganFingerprintAsBitVect
    pars = { "radius": 2,
                     "nBits": 65536,
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
    smiles = list(df['standard_smiles'])
    onbits_fp = list(df['onbits_fp'])
    zincid = list(df['canonical_id'])
    # pars = { "radius": 2,
    #          "nBits": 8192,
    #          "invariants": [],
    #          "fromAtoms": [],
    #          "useChirality": False,
    #          "useBondTypes": True,
    #          "useFeatures": False,
    #          }

    # fingerprint_function = rdMolDescriptors.GetMorganFingerprintAsBitVect        
    print(f'the number of smiles in the record batch is {len(smiles)}')
    count_ligs = len(smiles)



    # scores_list = []
    name_list =[]
    # smiles_list = []
    row_idx = list()
    col_idx = list()
    num_on_bits = []
    for count,m in enumerate(smiles):
    #     m_in = str(m)
    #     mol = Chem.MolFromSmiles(m_in)
    #     fp = fingerprint_function(mol, **pars)
    #     score = str(scores[count])
        zincid_name = str(zincid[count])
        onbits = list(onbits_fp[count])
    #     print(onbits)

    #     print(type(onbits))
        col_idx+=onbits
        row_idx += [count]*len(onbits)
        num_bits = len(onbits)
        num_on_bits.append(num_bits)
    #     scores_list.append(score)
        name_list.append(zincid_name)


        # except:
        #     print('molecule failed')

    unfolded_size = 65536        
    fingerprint_matrix = sparse.coo_matrix((np.ones(len(row_idx)).astype(bool), (row_idx, col_idx)), 
              shape=(max(row_idx)+1, unfolded_size))
    fingerprint_matrix =  sparse.csr_matrix(fingerprint_matrix)
    fp_mat = fingerprint_matrix
    print('Fingerprint matrix shape:', fp_mat.shape)
    print('\n')
    print('Indices:', fp_mat.indices)
    print('Indices shape:', fp_mat.indices.shape)
    print('\n')
    print('Index pointer:', fp_mat.indptr)
    print('Index pointer shape:', fp_mat.indptr.shape)
    print('\n')
    print('Actual data (these are all just "ON" bits!):', fp_mat.data)
    print('Actual data shape:', fp_mat.data.shape)
    return fp_mat

@stopwatch
def fast_dice(X, Y=None):
    if isinstance(X, np.ndarray):
        X = sparse.csr_matrix(X).astype(bool).astype(int)
    if Y is None:
        Y = X
    else:
        if isinstance(Y, np.ndarray):
            Y = sparse.csr_matrix(Y).astype(bool).astype(int)
            
    intersect = X.dot(Y.T)
    #cardinality = X.sum(1).A
    cardinality_X = X.getnnz(1)[:,None] #slightly faster on large matrices - 13s vs 16s for 12k x 12k
    cardinality_Y = Y.getnnz(1) #slightly faster on large matrices - 13s vs 16s for 12k x 12k
    return (1-(2*intersect) / (cardinality_X+cardinality_Y.T)).A

@stopwatch
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

@stopwatch
def _show_nearest_neighbor(i, row):

    """Use the output matrix from similarity search to find nearest neighbors from the
    reference matrix. out_array must be a globally defined variable for this to work.

    """
    a = out_array[i]




    minval = np.min(ma.masked_where(a==0, a)) 


#     maxval = np.max(ma.masked_where(a==0, a)) 



    minvalpos = np.argmin(ma.masked_where(a==0, a))  



#     maxvalpos = np.argmax(ma.masked_where(a==0, a))  

    
    smiles_nn = smiles[minvalpos]
    name_nn = name[minvalpos]



    row["nearest_neighbor_smiles"] = smiles_nn 
    row["nearest_neighbor_name"] = name_nn
    row["nearest_neighbor_distance"] = minval
    return row

@stopwatch
def ingest_chembl_smi(smi_path, smiles_column, canonical_id_column, activity_column):
    
    """Convert an smi file with a smiles column to a molchunk. It is assumed that
        the SMI has been cleaned (no header, and other columns have been removed).
        
    Args:
        smi_path: path to the smi file.
        smiles_column: column where the SMILES are located: f0 = col 1 f1 = col 2 .. etc
        canonical_id_column: name/id for molecule: f0 = col 1 f1 = col 2 .. etc
        activity column: column where bioactivity is listed (ki, ec50, etc): f0 = col 1 f1 = col 2 .. etc

    """

    
    # Next we will the multithreaded read options that pyarrow allows for.

    opts = pa.csv.ReadOptions(use_threads=True, autogenerate_column_names=True)

    # Then we tell pyarrow that the columns in our csv file are seperated by ';'
    # If they were tab seperated we would use '\t' and if it was comma we would use 
    # ','
    parse_options= pa.csv.ParseOptions(delimiter=' ')

    # Now we read the CSV into a pyarrow table. This is a columular dataset. More
    # on this later. Note how we specified the options above.

    table = pa.csv.read_csv(smi_path, opts, parse_options)


    # Now we will use a function that converts the pyarrow table into a pandas 
    # dataframe. We could have done this without arrow, but again -- there are 
    # very powerful tools that arrow will grant us.

    df_new = table.to_pandas()
 
    smiles_column = 'f0'
    
    # run initial mapper on smiles column to generate basic information and fingerprint on bits
    df_clean_mapped = dm.parallelized(_preprocess, list(df_new.iterrows()), arg_type='args', progress=True)
    df_clean_mapped = pd.DataFrame(df_clean_mapped)
    
    #rename columns
    df_clean_mapped['smiles'] = df_clean_mapped[smiles_column]
    df_clean_mapped['canonical_id'] = df_clean_mapped[canonical_id_column]
    df_clean_mapped['ki'] = df_clean_mapped[activity_column]
    
    #delete old columns
    del df_clean_mapped['f2']
    del df_clean_mapped['f1']
    del df_clean_mapped['f0']
    
    #remove duplicated standard SMILES and reindex
    duplicateRowsDF2 = df_clean_mapped[df_clean_mapped.duplicated(['standard_smiles'])]
    print("Duplicate Rows based on a single column are:", duplicateRowsDF2, sep='\n')
    df_clean_mapped = df_clean_mapped.drop_duplicates(subset='standard_smiles', keep="first", inplace=False)
    df = df_clean_mapped.reset_index(drop=True)
    
    return df

@stopwatch
def smiles_to_oemol(smiles,title='MOL'):
    """Create a OEMolBuilder from a smiles string.
    Parameters
    ----------
    smiles : str
        SMILES representation of desired molecule.
    Returns
    -------
    molecule : OEMol
        A normalized molecule with desired smiles string.
    """

    oechem = import_("openeye.oechem")
    if not oechem.OEChemIsLicensed(): raise(ImportError("Need License for OEChem!"))

    molecule = oechem.OEMol()
    if not oechem.OEParseSmiles(molecule, smiles):
        raise ValueError("The supplied SMILES '%s' could not be parsed." % smiles)

    molecule = normalize_molecule(molecule)

    # Set title.
    molecule.SetTitle(title)

    return molecule

@stopwatch
def generate_conformers(molecule, max_confs=800, strictStereo=True, ewindow=15.0, rms_threshold=1.0, strictTypes = True):
    """Generate conformations for the supplied molecule
    Parameters
    ----------
    molecule : OEMol
        Molecule for which to generate conformers
    max_confs : int, optional, default=800
        Max number of conformers to generate.  If None, use default OE Value.
    strictStereo : bool, optional, default=True
        If False, permits smiles strings with unspecified stereochemistry.
    strictTypes : bool, optional, default=True
        If True, requires that Omega have exact MMFF types for atoms in molecule; otherwise, allows the closest atom type of the same element to be used.
    Returns
    -------
    molcopy : OEMol
        A multi-conformer molecule with up to max_confs conformers.
    Notes
    -----
    Roughly follows
    http://docs.eyesopen.com/toolkits/cookbook/python/modeling/am1-bcc.html
    """
#     os.environ["OE_LICENSE"] = "/data/openeye/oe_license.txt"
    oechem = import_("openeye.oechem")
    if not oechem.OEChemIsLicensed(): raise(ImportError("Need License for OEChem!"))
    oeomega = import_("openeye.oeomega")
    if not oeomega.OEOmegaIsLicensed(): raise(ImportError("Need License for OEOmega!"))

    molcopy = oechem.OEMol(molecule)
    omega = oeomega.OEOmega()

    # These parameters were chosen to match http://docs.eyesopen.com/toolkits/cookbook/python/modeling/am1-bcc.html
    omega.SetMaxConfs(max_confs)
    omega.SetIncludeInput(True)
    omega.SetCanonOrder(False)

    omega.SetSampleHydrogens(True)  # Word to the wise: skipping this step can lead to significantly different charges!
    omega.SetEnergyWindow(ewindow)
    omega.SetRMSThreshold(rms_threshold)  # Word to the wise: skipping this step can lead to significantly different charges!

    omega.SetStrictStereo(strictStereo)
    omega.SetStrictAtomTypes(strictTypes)

    omega.SetIncludeInput(False)  # don't include input
    if max_confs is not None:
        omega.SetMaxConfs(max_confs)

    status = omega(molcopy)  # generate conformation
    if not status:
        raise(RuntimeError("omega returned error code %d" % status))


    return molcopy

@stopwatch
def normalize_molecule(molecule):
    """
    Normalize a copy of the molecule by checking aromaticity, adding explicit hydrogens, and
    (if possible) renaming by IUPAC name.
    Parameters
    ----------
    molecule : OEMol
        the molecule to be normalized.
    Returns
    -------
    molcopy : OEMol
        A (copied) version of the normalized molecule
    """
#     os.environ["OE_LICENSE"] = "/data/openeye/oe_license.txt"
    oechem = import_("openeye.oechem")
    if not oechem.OEChemIsLicensed():
        raise(ImportError("Need License for OEChem!"))
    oeiupac = import_("openeye.oeiupac")
    has_iupac = oeiupac.OEIUPACIsLicensed()

    molcopy = oechem.OEMol(molecule)

    # Assign aromaticity.
    oechem.OEAssignAromaticFlags(molcopy, oechem.OEAroModelOpenEye)

    # Add hydrogens.
    oechem.OEAddExplicitHydrogens(molcopy)

    # Set title to IUPAC name.
    if has_iupac:
        name = oeiupac.OECreateIUPACName(molcopy)
        molcopy.SetTitle(name)

    # Check for any missing atom names, if found reassign all of them.
    if any([atom.GetName() == '' for atom in molcopy.GetAtoms()]):
        oechem.OETriposAtomNames(molcopy)

    return molcopy

@stopwatch
def molecule_to_mol2(molecule, tripos_mol2_filename=None, conformer=0, residue_name="MOL", standardize=True):
    """Convert OE molecule to tripos mol2 file.
    Parameters
    ----------
    molecule : openeye.oechem.OEGraphMol
        The molecule to be converted.
    tripos_mol2_filename : str, optional, default=None
        Output filename.  If None, will create a filename similar to
        name.tripos.mol2, where name is the name of the OE molecule.
    conformer : int, optional, default=0
        Save this frame
        If None, save all conformers
    residue_name : str, optional, default="MOL"
        OpenEye writes mol2 files with <0> as the residue / ligand name.
        This chokes many mol2 parsers, so we replace it with a string of
        your choosing.
    standardize: bool, optional, default=True
        Use a high-level writer, which will standardize the molecular properties.
        Set this to false if you wish to retain things such as atom names.
        In this case, a low-level writer will be used.
    Returns
    -------
    tripos_mol2_filename : str
        Filename of output tripos mol2 file
    """
#     os.environ["OE_LICENSE"] = "/data/openeye/oe_license.txt"
    oechem = import_("openeye.oechem")
    if not oechem.OEChemIsLicensed(): raise(ImportError("Need License for oechem!"))

    # Get molecule name.
    molecule_name = molecule.GetTitle()
    logger.debug(molecule_name)

    # Write molecule as Tripos mol2.
    if tripos_mol2_filename is None:
        tripos_mol2_filename = molecule_name + '.tripos.mol2'

    ofs = oechem.oemolostream(tripos_mol2_filename)
    ofs.SetFormat(oechem.OEFormat_MOL2H)
    for k, mol in enumerate(molecule.GetConfs()):
        if k == conformer or conformer is None:
            # Standardize will override molecular properties(atom names etc.)
            if standardize:
                oechem.OEWriteMolecule(ofs, mol)
            else:
                oechem.OEWriteMol2File(ofs, mol)

    ofs.close()

    # Replace <0> substructure names with valid text.
    infile = open(tripos_mol2_filename, 'r')
    lines = infile.readlines()
    infile.close()
    newlines = [line.replace('<0>', residue_name) for line in lines]
    outfile = open(tripos_mol2_filename, 'w')
    outfile.writelines(newlines)
    outfile.close()

    return molecule_name, tripos_mol2_filename

@stopwatch
def get_charges(molecule, max_confs=800, strictStereo=True,
                normalize=True, keep_confs=None, legacy=False):
    """Generate charges for an OpenEye OEMol molecule.
    Parameters
    ----------
    molecule : OEMol
        Molecule for which to generate conformers.
        Omega will be used to generate max_confs conformations.
    max_confs : int, optional, default=800
        Max number of conformers to generate
    strictStereo : bool, optional, default=True
        If False, permits smiles strings with unspecified stereochemistry.
        See https://docs.eyesopen.com/omega/usage.html
    normalize : bool, optional, default=True
        If True, normalize the molecule by checking aromaticity, adding
        explicit hydrogens, and renaming by IUPAC name.
    keep_confs : int, optional, default=None
        If None, apply the charges to the provided conformation and return
        this conformation, unless no conformation is present.
        Otherwise, return some or all of the generated
        conformations. If -1, all generated conformations are returned.
        Otherwise, keep_confs = N will return an OEMol with up to N
        generated conformations.  Multiple conformations are still used to
        *determine* the charges.
    legacy : bool, default=True
        If False, uses the new OpenEye charging engine.
        See https://docs.eyesopen.com/toolkits/python/quacpactk/OEProtonFunctions/OEAssignCharges.html#
    Returns
    -------
    charged_copy : OEMol
        A molecule with OpenEye's recommended AM1BCC charge selection scheme.
    Notes
    -----
    Roughly follows
    http://docs.eyesopen.com/toolkits/cookbook/python/modeling/am1-bcc.html
    """
#     os.environ["OE_LICENSE"] = "/data/openeye/oe_license.txt"
    # If there is no geometry, return at least one conformation.
    if molecule.GetConfs() == 0:
        keep_confs = 1

    oechem = import_("openeye.oechem")
    if not oechem.OEChemIsLicensed(): raise(ImportError("Need License for OEChem!"))
    oequacpac = import_("openeye.oequacpac")
    if not oequacpac.OEQuacPacIsLicensed(): raise(ImportError("Need License for oequacpac!"))

    if normalize:
        molecule = normalize_molecule(molecule)
    else:
        molecule = oechem.OEMol(molecule)

    charged_copy = generate_conformers(molecule, max_confs=max_confs, strictStereo=strictStereo)  # Generate up to max_confs conformers

    if not legacy:
        # try charge using AM1BCCELF10
        status = oequacpac.OEAssignCharges(charged_copy, oequacpac.OEAM1BCCELF10Charges())
        # or fall back to OEAM1BCC
        if not status:
            # 2017.2.1 OEToolkits new charging function
            status = oequacpac.OEAssignCharges(charged_copy, oequacpac.OEAM1BCCCharges())
            if not status:
                # Fall back
                status = oequacpac.OEAssignCharges(charged_copy, oequacpac.OEAM1Charges())

                # Give up
                if not status:
                    raise(RuntimeError("OEAssignCharges failed."))
    else:
        # AM1BCCSym recommended by Chris Bayly to KAB+JDC, Oct. 20 2014.
        status = oequacpac.OEAssignPartialCharges(charged_copy, oequacpac.OECharges_AM1BCCSym)
        if not status: raise(RuntimeError("OEAssignPartialCharges returned error code %d" % status))

    #Determine conformations to return
    if keep_confs == None:
        #If returning original conformation
        original = molecule.GetCoords()
        #Delete conformers over 1
        for k, conf in enumerate( charged_copy.GetConfs() ):
            if k > 0:
                charged_copy.DeleteConf(conf)
        #Copy coordinates to single conformer
        charged_copy.SetCoords( original )
    elif keep_confs > 0:
        logger.debug("keep_confs was set to %s. Molecule positions will be reset." % keep_confs)

        #Otherwise if a number is provided, return this many confs if available
        for k, conf in enumerate( charged_copy.GetConfs() ):
            if k > keep_confs - 1:
                charged_copy.DeleteConf(conf)
    elif keep_confs == -1:
        #If we want all conformations, continue
        pass
    else:
        #Not a valid option to keep_confs
        raise(ValueError('Not a valid option to keep_confs in get_charges.'))

    return charged_copy

@stopwatch
def get_mol2_string_from_OEMol(molecule):
#     os.environ["OE_LICENSE"] = "/data/openeye/oe_license.txt"
    oechem = import_("openeye.oechem")
    if not oechem.OEChemIsLicensed(): raise(ImportError("Need License for OEChem!"))
    molecule_name = molecule.GetTitle()
    conformer=0
    standardize=True
#     print(molecule.GetConfs())



    ofs = oechem.oemolostream()
    ofs.SetFormat(oechem.OEFormat_MOL2H)
    ofs.openstring()
    for k, mol in enumerate(molecule.GetConfs()):
        if k == conformer or conformer is None:
            # Standardize will override molecular properties(atom names etc.)
            if standardize:
                oechem.OEWriteMolecule(ofs, mol)
            else:
                oechem.OEWriteMol2File(ofs, mol)

    molfile = ofs.GetString()
    return molfile

@stopwatch
def df_from_dude_smi_file(dude_smi_file, delimiter, name_col, smi_col):
    parse_options = pa.csv.ParseOptions(delimiter=delimiter)
    read_options = pa.csv.ReadOptions(use_threads=True)
    table = csv.read_csv(dude_smi_file, parse_options=parse_options)
    name_col = int(name_col)
    smi_col = int(smi_col)
    name_arr = table.column(name_col)
    smiles_arr = table.column(smi_col)

    data = [
        name_arr,
        smiles_arr
    ]

    table2 = pa.Table.from_arrays(data, names=['names', 'smiles'])
    df = pa.Table.to_pandas(table2)
    return df


@stopwatch
def get_3d_from_datamol(smiles, names, output_dir, count):
    print(f'smiles prior to sanitize_smiles : {smiles}')
    smiles= mol.sanitize_smiles(smiles)
    print(f'smiles after sanitize_smiles : {smiles}')
    mol = dm.to_mol(smiles)

@stopwatch
def get_3d_from_corina(smiles, names, output_dir, count):
    print(f'smiles prior to sanitize_smiles : {smiles}')
    smiles= mol.sanitize_smiles(smiles)
    print(f'smiles after sanitize_smiles : {smiles}')
    smiles_list = [smiles]
    names_list = [names]
    failed_smiles = pd.DataFrame(list(zip(smiles_list, names_list)), \
                             columns = ['smiles', 'names'])
    temp = f'{output_dir}/temp.smiles'
    failed_smiles.to_csv(temp, sep=' ', header=False, index=False)
    out_sdf_path = f'{output_dir}/temp3d_{count}.sdf'
    args = f'/home/schrogpu/corina/corina -i t=smiles,sep=" ",scn=1,ncn=1,ccn=3 {temp} {out_sdf_path}'
#     print(args)
    x = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    std_out, std_err = x.communicate()
    if std_err :
        print(std_err)

    else:
        molecule = oe_sdf_to_molecule(out_sdf_path)
        
    
#     print(type(molecule))    
    return molecule

@stopwatch        
def oe_sdf_to_molecule(out_sdf_path):
    oechem = import_("openeye.oechem")
    if not oechem.OEChemIsLicensed(): raise(ImportError("Need License for OEChem!"))
    ifs = oechem.oemolistream()
    ifs.SetFormat(oechem.OEFormat_SDF)
    oms = oechem.oemolostream()
    oms.SetFormat(oechem.OEFormat_SDF)
    oms.openstring()
    
    mols = []
    mol = oechem.OEMol()
    if ifs.open(out_sdf_path):
        for mol in ifs.GetOEGraphMols():
            mols.append(oechem.OEMol(mol))

    else:
        oechem.OEThrow.Fatal(f"Unable to open {out_sdf_path}")
#     print(type(mols[0]))
    try:
        molecule = mols[0]
        for mol in mols:
            oechem.OEWriteMolecule(oms, mol)

        molfile = oms.GetString()
        print("MOL string\n", molfile.decode('UTF-8'))
        return molecule
    except IndexError:
        print('index error, failed to generate conformers')
        return None



def _preprocess_3d(i, row):
#     print('hello')
    try:
        smiles = str(row[smiles_column])
        name = row[6]
#         print(f'Converting {name} with smiles {smiles} to PDBQT')
        mol = smiles_to_oemol(smiles,title=name)
        mol3 = get_charges(mol, max_confs=800, strictStereo=True,
        normalize=True, keep_confs=None, legacy=False)
        mol2_block = get_mol2_string_from_OEMol(mol3)
        mol2_block_string = mol2_block.decode("utf-8")
        mol2_pybel = pybel.readstring(format='mol2', string=mol2_block_string)

        pdbqt_am1bcc = mol2_pybel.write(format='pdbqt')

        mol2_block_am1bcc = mol2_block
        pdb_am1bcc = mol2_pybel.write(format='pdb')

        mol2_pybel.calccharges(model='gasteiger')
        pdbqt_gast = mol2_pybel.write(format='pdbqt')


        row["mol2_block_am1bcc"] = mol2_block_am1bcc
        row["pdb_am1bcc"] = pdb_am1bcc
        row["pdbqt_am1bcc"] = pdbqt_am1bcc
        row["pdbqt_gast"] = pdbqt_gast
#         print(f'{name} with smiles {smiles} is complete')
        return row
    except:
        smiles = str(row[smiles_column])
        name = row[6]


        row["mol2_block_am1bcc"] = 'dropped'
        row["pdb_am1bcc"] = 'dropped'
        row["pdbqt_am1bcc"] = 'dropped'
        row["pdbqt_gast"] = 'dropped'
#         print(f'{name} with smiles {smiles} is failed!!')
        return row

smiles_column = 'standard_smiles'
df_clean_mapped_3d = dm.parallelized(_preprocess_3d, list(d3_df.iterrows()), arg_type='args', progress=True)
df_clean_mapped_3d_1 = pd.DataFrame(df_clean_mapped_3d)
df2 = df_clean_mapped_3d_1

df2 = df2.set_index('canonical_id')
df2.index = df2.index + df2.groupby(level=0).cumcount().astype(str).replace('0','')
df2 = df2.reset_index()
df2['canonical_id'] = df2['index']
del df2['index']
df2
feather.write_feather(df_clean_mapped_3d_1, '/data/dockop_glide_d3/chembld3.molchunk')
