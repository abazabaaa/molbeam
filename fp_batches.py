
import sys
import duckdb
from io import StringIO
import pyarrow as pa
from pyarrow import csv
import pandas as pd
import pyarrow.feather as feather
from rdkit.Chem import PandasTools

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import Draw
from rdkit.Chem import DataStructs

import numpy as np
import numpy.ma as ma
import tqdm
import pyarrow.parquet as pq
import numpy as np
from scipy import sparse
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
import pyarrow.parquet as pq
from tqdm.contrib import tenumerate
from tqdm import tqdm

import pyarrow.feather as fe
import datamol as dm
import operator

from datetime import timedelta
from timeit import time



from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA

rdDepictor.SetPreferCoordGen(True)
from bokeh.plotting import ColumnDataSource, figure, output_notebook, show
from bokeh.io import output_notebook
output_notebook()

dm.disable_rdkit_log()
#autocomplete wasn't working for some reason. This fixes it. 
%config Completer.use_jedi = False


def stopwatch(method):
    def timed(*args, **kw):
        ts = time.perf_counter()
        result = method(*args, **kw)
        te = time.perf_counter()
        duration = timedelta(seconds=te - ts)
        print(f"{method.__name__}: {duration}")
        return result
    return timed


def generateconformations(m):
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m)
    m = Chem.RemoveHs(m)
    return m    



def mol_to_xyz(mol_w_confs):
    number_of_atoms = mol_w_confs.GetNumAtoms()
    symbols = [a.GetSymbol() for a in mol_w_confs.GetAtoms()]
    f = StringIO()

    f.write(str(number_of_atoms)+"\n")
    f.write("title\n")
    conf = mol_w_confs.GetConformers()[0]
    for atom,symbol in enumerate(symbols):
        p = conf.GetAtomPosition(atom)
        line = " ".join((symbol,str(p.x),str(p.y),str(p.z),"\n"))
        f.write(line)

    xyz = f.getvalue()
    f.close()
    return xyz

def _preprocess(i, row):
#     print('hello')
    mol = dm.to_mol(str(row[smiles_column]), ordered=True)
    mol = dm.fix_mol(mol)
    mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
    

    mol = dm.standardize_mol(mol, disconnect_metals=False, normalize=True, reionize=True, uncharge=False, stereo=True)
    
    mol_w_confs = generateconformations(mol)
    xyz = mol_to_xyz(mol_w_confs)

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
    row["xyz"] = xyz
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
    # pars = { "radius": 2,
    #          "nBits": 8192,
    #          "invariants": [],
    #          "fromAtoms": [],
    #          "useChirality": False,
    #          "useBondTypes": True,
    #          "useFeatures": False,
    #          }

    # fingerprint_function = rdMolDescriptors.GetMorganFingerprintAsBitVect        
#     print(f'the number of smiles in the record batch is {len(smiles)}')
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

    unfolded_size = 8192        
    fingerprint_matrix = sparse.coo_matrix((np.ones(len(row_idx)).astype(bool), (row_idx, col_idx)), 
              shape=(max(row_idx)+1, unfolded_size))
    fingerprint_matrix =  sparse.csr_matrix(fingerprint_matrix)
    fp_mat = fingerprint_matrix
#     print('Fingerprint matrix shape:', fp_mat.shape)
#     print('\n')
#     print('Indices:', fp_mat.indices)
#     print('Indices shape:', fp_mat.indices.shape)
#     print('\n')
#     print('Index pointer:', fp_mat.indptr)
#     print('Index pointer shape:', fp_mat.indptr.shape)
#     print('\n')
#     print('Actual data (these are all just "ON" bits!):', fp_mat.data)
#     print('Actual data shape:', fp_mat.data.shape)
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
#     print("Duplicate Rows based on a single column are:", duplicateRowsDF2, sep='\n')
    df_clean_mapped = df_clean_mapped.drop_duplicates(subset='standard_smiles', keep="first", inplace=False)
    df = df_clean_mapped.reset_index(drop=True)
    
    return df

def _preprocess_dwar(i, row):
#     print('hello')
    try:
        mol = dm.to_mol(str(row[smiles_column]), ordered=True)
        mol = dm.fix_mol(mol)
        mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)


        mol = dm.standardize_mol(mol, disconnect_metals=False, normalize=True, reionize=True, uncharge=False, stereo=True)
        mol_w_confs = generateconformations(mol)
        xyz = mol_to_xyz(mol_w_confs)
        
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
        row['xyz'] = xyz
        row["enumerated_smiles"] = dm.standardize_smiles(dm.to_smiles(mol))
        row["selfies"] = dm.to_selfies(mol)
        row["inchi"] = dm.to_inchi(mol)
        row["inchikey"] = dm.to_inchikey(mol)
        row["achiral_fp"] =list(fp.GetOnBits())
        
        return row
        
    except ValueError:
        row["enumerated_smiles"] = 'dropped'
        row["selfies"] = 'dropped'
        row["inchi"] = 'dropped'
        row["inchikey"] = 'dropped'
        row["achiral_fp"] = 'dropped'
        
        return row

@stopwatch
def ingest_chembl_raw_csv(csv_path):
    
    """Convert a csv file from chembl into a molchunk that has been processed with datamol.
        
    Args:
        csv_path: path to the csv file.


    """

    
    # Next we will the multithreaded read options that pyarrow allows for.

    opts = pa.csv.ReadOptions(use_threads=True)

    # Then we tell pyarrow that the columns in our csv file are seperated by ';'
    # If they were tab seperated we would use '\t' and if it was comma we would use 
    # ','
    parse_options= pa.csv.ParseOptions(delimiter=';')

    # Now we read the CSV into a pyarrow table. This is a columular dataset. More
    # on this later. Note how we specified the options above.

    table = pa.csv.read_csv(csv_path, opts, parse_options)


    #clean up the column names and convert the activity column to ki
    
    

    df_new = table.to_pandas()
    df_new['ki'] = df_new[['Standard Value']].astype(str).agg(''.join, axis=1)
    df_clean = df_new[['Molecule ChEMBL ID', 'ki', 'Smiles']]
    df_clean = df_clean.rename({'Molecule ChEMBL ID':'canonical_id', 'ki':'ki', 'Smiles':'smiles'}, axis='columns')
    df_clean['length'] = df_clean.smiles.str.len()
    df_clean = df_clean[df_clean.length > 0]
    del df_clean['length']

 
#     smiles_column = 'smiles'
    
#     run initial mapper on smiles column to generate basic information and fingerprint on bits
    df_clean_mapped = dm.parallelized(_preprocess, list(df_clean.iterrows()), arg_type='args', progress=True)
    df_clean_mapped = pd.DataFrame(df_clean_mapped)
    
    
    #remove duplicated standard SMILES and reindex
    duplicateRowsDF2 = df_clean_mapped[df_clean_mapped.duplicated(['standard_smiles'])]
    print("Duplicate Rows based on a single column are:", duplicateRowsDF2, sep='\n')
    df_clean_mapped = df_clean_mapped.drop_duplicates(subset='standard_smiles', keep="first", inplace=False)
    df_clean_mapped = df_clean_mapped.reset_index(drop=True)
    
    return df_clean_mapped


def read_tab_delimited_dwar(path):
#     csv_path = '/Users/tom/Downloads/Screened_Compound_Updated-3.txt'

    opts = pa.csv.ReadOptions(use_threads=True)

    # Then we tell pyarrow that the columns in our csv file are seperated by ';'
    # If they were tab seperated we would use '\t' and if it was comma we would use 
    # ','
    parse_options= pa.csv.ParseOptions(delimiter='\t')

    # Now we read the CSV into a pyarrow table. This is a columular dataset. More
    # on this later. Note how we specified the options above.

    table = pa.csv.read_csv(path, opts, parse_options)
    return table

def _preprocess_addpartition_col_ray(count, row):
## This is hardcoded for now and should be fixed
    index = int(row[8])
    assigned_batch = math.ceil(index/rows_per_batch)
    if math.ceil(index/rows_per_batch) == 0:
        assigned_batch = 1

#     print(f'column number {count} finished')
    row["partition_col"] = assigned_batch 
    return row

from rdkit import Chem
import warnings

def has_all_chiral_defined_rd(smiles):
    try:
        undefined_atoms = []
        unspec_chiral = False
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        for center in chiral_centers:
            atom_id = center[0]
            if center[-1] == '?':
                unspec_chiral = True
                undefined_atoms.append((atom_id, mol.GetAtomWithIdx(atom_id).GetSmarts()))
        if unspec_chiral:
            print(undefined_atoms)
            return False
        else:
            return True
    except:
        return False

def has_stereo_defined(molecule):
    """

    Parameters
    ----------
    molecule

    Returns
    -------

    """

    unspec_chiral = False
    unspec_db = False
    problematic_atoms = list()
    problematic_bonds = list()
    chiral_centers = Chem.FindMolChiralCenters(molecule, includeUnassigned=True)
    for center in chiral_centers:
        atom_id = center[0]
        if center[-1] == '?':
            unspec_chiral = True
            problematic_atoms.append((atom_id, molecule.GetAtomWithIdx(atom_id).GetSmarts()))

    # Find potential stereo bonds that are unspecified
    Chem.FindPotentialStereoBonds(molecule)
    for bond in molecule.GetBonds():
        if bond.GetStereo() == Chem.BondStereo.STEREOANY:
            unspec_db = True
            problematic_bonds.append((bond.GetBeginAtom().GetSmarts(), bond.GetSmarts(),
                                                bond.GetEndAtom().GetSmarts()))
    if unspec_chiral or unspec_db:
        warnings.warn("Stereochemistry is unspecified. Problematic atoms {}, problematic bonds {}".format(
                problematic_atoms, problematic_bonds))
        return False
    else:
        return True
    
    
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

def ingest_datawarrior_synuclein(table):
    df = table.to_pandas()
    cols_to_keep = ['Name', 'SMILES', 'Source', 'Ki (nM), site 9', "% Inh, 1000 nM, S9", "% Inh, 100nM, S9", "% Inh, 10 nM, S9", "%Inh 1 uM, S2", "%inh 100 nM, S2", "%inh, 10 nM, S2", "Tris fibrils, site 2, IC50 (nM)", "PBS fibrils, site 2 (IC50, nM)", "%Inh, 1000 nM, PiB", '%Inh, 100 nM, PiB', '%Inh, 10 nM, PiB', 'Ki (nM) in AD vs PiB']
    df2 = df[cols_to_keep]
    nan_value = float("NaN")
    df2.replace("", nan_value, inplace=True)
    df2.dropna(subset = ["SMILES"], inplace=True)
    df3 = df2.reset_index(drop=True)
    
    #     run initial mapper on smiles column to generate basic information and fingerprint on bits
    df_clean_mapped = dm.parallelized(_preprocess_dwar, list(df3.iterrows()), arg_type='args', progress=True)
    df_clean_mapped = pd.DataFrame(df_clean_mapped)
    word = 'dropped'
    df_clean_mapped2 = df_clean_mapped[~df_clean_mapped["enumerated_smiles"].str.contains(word)]
    df_clean_mapped3 = df_clean_mapped2.rename(columns={'Name': 'canonical_ID'})
    df_clean_mapped4 = df_clean_mapped3.reset_index(drop=True)
    data = df_clean_mapped4
    return data

def ingest_fastrocs_dwar(table2):
    df_pbb3 = table2.to_pandas()
    cols_to_keep = ['IDNUMBER', 'Smile', 'ROCS_TanimotoCombo', 'CNS-MPO']
    df2_pbb3 = df_pbb3[cols_to_keep]
    df_clean_mapped = dm.parallelized(_preprocess_dwar, list(df2_pbb3.iterrows()), arg_type='args', progress=True)
    df_clean_mapped = pd.DataFrame(df_clean_mapped)
    word = 'dropped'
    df_clean_mapped2 = df_clean_mapped[~df_clean_mapped["enumerated_smiles"].str.contains(word)]
    df_clean_mapped3 = df_clean_mapped2.rename(columns={'IDNUMBER': 'canonical_ID'})
    df_clean_mapped4 = df_clean_mapped3.reset_index(drop=True)
    data = df_clean_mapped4
    return data

def table_to_df_mols_comparison(df): 
    combined_df = df

    PandasTools.AddMoleculeColumnToFrame(combined_df, smilesCol='enumerated_smiles', molCol='self')
    PandasTools.AddMoleculeColumnToFrame(combined_df, smilesCol='nn_smiles', molCol='near_neighbor')
    PandasTools.RenderImagesInAllDataFrames()

    combined_df = combined_df.reset_index(drop=True)
    cols_to_keep = ['canonical_ID', 'ROCS_TanimotoCombo', 'CNS-MPO', 'self', 'near_neighbor', 'nn_smiles', 'nn_name', 'nn_distance', 'nn_sumval', 'enumerated_smiles']
    combined_df = combined_df[cols_to_keep]
    return combined_df


def table_to_df_mols_comparison_pbb3(df): 
    combined_df = df

    PandasTools.AddMoleculeColumnToFrame(combined_df, smilesCol='enumerated_smiles', molCol='self')
    PandasTools.AddMoleculeColumnToFrame(combined_df, smilesCol='nn_smiles', molCol='near_neighbor')
    PandasTools.RenderImagesInAllDataFrames()

    combined_df = combined_df.reset_index(drop=True)
    cols_to_keep = ['canonical_ID', 'self', 'near_neighbor', 'nn_smiles', 'nn_name', 'nn_distance', 'enumerated_smiles']
    combined_df = combined_df[cols_to_keep]
    return combined_df

def mol2svg(mol):
    AllChem.Compute2DCoords(mol)
    d2d = rdMolDraw2D.MolDraw2DSVG(200,100)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()

def mol2fparr(mol):
    arr = np.zeros((0,))
    fingerprint_function = rdMolDescriptors.GetMorganFingerprintAsBitVect
    pars2 = { "radius": 2,
                 "nBits": 8192,
                 "invariants": [],
                 "fromAtoms": [],
                 "useChirality": False,
                 "useBondTypes": True,
                 "useFeatures": False, }
    fp = fingerprint_function(mol, **pars2)
#     fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def pca_on_df(query_df):
    mols2 = [dm.to_mol(x) for x in list(query_df['nn_smiles'])]
    mols3 = [dm.to_mol(x) for x in list(query_df['enumerated_smiles'])]

    fps = np.array([mol2fparr(m) for m in mols2])
    pca = PCA(n_components=15)
    chemicalspace = pca.fit_transform(fps)
    kmean = KMeans(n_clusters=4, random_state=32)
    kmean.fit(fps)
    colormaps = {0:'red',1:'blue',2:'green',3:'orange',4:'yellow',5:'violet',6:'teal',7:'black',8:'gray'}
    TOOLTIPS = """
    <div>
    name: @ids<br>

    nn_distance: @nn_distance<br>

    @img{safe}
    MolID: @ids2<br>
    @img2{safe}
    </div>
    """

    structure_name = list(query_df['nn_name'])
    structure_name_nn = list(query_df['canonical_ID'])
#     rocs_score = list(query_df['ROCS_TanimotoCombo'])
    nn_distances = list(query_df['nn_distance'])
#     nn_sumvals = list(query_df['nn_sumval'])
    # # Make an image from the molecules list with their SMILES as legend.
    # legends = [_ for _ in structure_name]



    kmeanc = [colormaps[i] for i in kmean.labels_]


    kmean_data = dict(
    x=chemicalspace[:,0],
    y=chemicalspace[:,1],
    ids=[_ for _ in structure_name],
    ids2=[_ for _ in structure_name_nn],
#     rocs=[_ for _ in rocs_score],
    nn_distance=[_ for _ in nn_distances],
#     nn_sumval=[_ for _ in nn_sumvals],
    img=[mol2svg(m) for m in mols2],
    img2=[mol2svg(m) for m in mols3],
    fill_color=kmeanc,
    )


    source = ColumnDataSource(kmean_data)
    p = figure(plot_width=900, plot_height=600, tooltips=TOOLTIPS,)
    p.circle('x', 'y',color='fill_color', size=10, fill_alpha=0.2,source=source)
    show(p)

    # #ingest the datawarrior synuclein file into a molchunk
# %time
# smiles_column = 'SMILES'
# path = '/Users/tom/Downloads/Screened_Compound_Updated-3.txt'
# table = read_tab_delimited_dwar(path)
# df = ingest_datawarrior_synuclein(table)    
# syn_database = fingerprint_matrix_from_df(df)
%time table_pq = pq.read_table('/Users/tom/U19/batch0007/testing/batch0014/er_real_35_to_end_2.parquet')
%time df_pq = table_pq.to_pandas()

# fingerprint_matrix_in = fingerprint_matrix_from_df(df_pq)
%time table_pq2 = pq.read_table('/Users/tom/U19/batch0007/testing/batch0014/er_real_35_to_end_2.parquet', columns=['canonical_ID', 'enumerated_smiles', 'achiral_fp'])

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
  
  data = make a list of lists

# Create the pandas DataFrame
df = pd.DataFrame(data, columns = ['smiles', 'canonical_ID'])

smiles_column = 'smiles'

# run initial mapper on smiles column to generate basic information and fingerprint on bits
df_clean_mapped = dm.parallelized(_preprocess, list(df.iterrows()), arg_type='args', progress=True)
df_clean_mapped = pd.DataFrame(df_clean_mapped)
df_clean_mapped['enumerated_smiles'] = df_clean_mapped['standard_smiles']
df_clean_mapped['achiral_fp'] = df_clean_mapped['onbits_fp']
fingerprint_matrix_pbb3 = fingerprint_matrix_from_df(df_clean_mapped)

#run the fast jaccard and compare all ligands in our library to 
# to each hit from the fastrocs screen
y = table_pq2.column('canonical_ID')
name = [pa.StringScalar.as_py(y[_]) for _ in range(len(y))]
z = table_pq2.column('enumerated_smiles')
smiles = [pa.StringScalar.as_py(z[_]) for _ in range(len(z))]

# smiles = list(df['enumerated_smiles'])
# name = list(df['canonical_ID'])

%time out_array = fast_jaccard(fingerprint_matrix, fingerprint_matrix_pbb3)
%time df_clean_mapped[['nn_smiles', 'nn_name', 'nn_distance']] = df_clean_mapped.apply(min_distance, axis=1, result_type='expand')

dataset = ds.dataset('/Users/tom/U19/batch0007/testing', format="parquet")
fragments = [file for file in dataset.get_fragments()]
total_frags = len(fragments)
for count,record_batch in tenumerate(dataset.to_batches(columns=["canonical_ID", "enumerated_smiles", "achiral_fp"]), start=0, total=total_frags):
    

        fingerprint_matrix_in = table_to_csr_fp(record_batch)
        y = table_from_pq.column('canonical_ID')
        name = [pa.StringScalar.as_py(y[_]) for _ in range(len(y))]
        z = table_from_pq.column('enumerated_smiles')
        smiles = [pa.StringScalar.as_py(z[_]) for _ in range(len(z))]
        out_array = fast_jaccard(fingerprint_matrix_in, fingerprint_matrix_pbb3)
        df_clean_mapped[['nn_smiles', 'nn_name', 'nn_distance']] = df_clean_mapped.apply(min_distance, axis=1, result_type='expand')
        cols_to_keep = ['canonical_ID', 'nn_smiles', 'nn_name', 'nn_distance', 'enumerated_smiles']
        df_clean_mapped = df_clean_mapped[cols_to_keep]
        table = pa.Table.from_pandas(df_clean_mapped, preserve_index=False)
        pq.write_table(table, f'/Users/tom/U19/batch0007/outchunks/comparison_{count}.parquet')

combined_df = df
PandasTools.AddMoleculeColumnToFrame(combined_df, smilesCol='enumerated_smiles', molCol='self')
PandasTools.AddMoleculeColumnToFrame(combined_df, smilesCol='nn_smiles', molCol='near_neighbor')
PandasTools.RenderImagesInAllDataFrames()
combined_df = combined_df.reset_index(drop=True)
cols_to_keep = ['canonical_ID', 'nn_name', 'nn_smiles', 'nn_distance', 'nn_distance', 'self', 'near_neighbor']
combined_df = combined_df[cols_to_keep]
combined_df        
