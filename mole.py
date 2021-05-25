import datamol as dm
from rdkit.Chem import rdMolDescriptors
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit import DataStructs
# from rdkit.Chem import rdMolDescriptors

dm.disable_rdkit_log()


def smiles_to_fingerprint(smiles):

    mol = dm.to_mol(str(smiles), ordered=True)
    # mol = dm.fix_mol(mol)
    # mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
    # mol = dm.standardize_mol(mol, disconnect_metals=False, normalize=True, reionize=True, uncharge=False, stereo=True)

    fingerprint_function = rdMolDescriptors.GetMorganFingerprintAsBitVect
    pars = {
        "radius": 2,
        "nBits": 8192,
        "invariants": [],
        "fromAtoms": [],
        "useChirality": False,
        "useBondTypes": True,
        "useFeatures": False,
    }
    fp = fingerprint_function(mol, **pars)

    standard_smiles = dm.to_smiles(mol)
    # row["selfies"] = dm.to_selfies(mol)
    # row["inchi"] = dm.to_inchi(mol)
    # row["inchikey"] = dm.to_inchikey(mol)
    achiral_fp = list(fp.GetOnBits())
    return standard_smiles, achiral_fp

# def normalize_smiles(item):

#     mol = dm.to_mol(str(item[0]), ordered=True)
#     mol = dm.fix_mol(mol)
#     mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
#     mol = dm.standardize_mol(mol, disconnect_metals=False, normalize=True, reionize=True, uncharge=False, stereo=True)

#     fingerprint_function = rdMolDescriptors.GetMorganFingerprintAsBitVect
#     pars = {
#         "radius": 2,
#         "nBits": 8192,
#         "invariants": [],
#         "fromAtoms": [],
#         "useChirality": False,
#         "useBondTypes": True,
#         "useFeatures": False,
#     }
#     fp = fingerprint_function(mol, **pars)

#     standard_smiles = dm.standardize_smiles(dm.to_smiles(mol))
#     # row["selfies"] = dm.to_selfies(mol)
#     # row["inchi"] = dm.to_inchi(mol)
#     # row["inchikey"] = dm.to_inchikey(mol)
#     achiral_fp = list(fp.GetOnBits())
#     query_name = item[1]

#     return standard_smiles, achiral_fp, query_name


# def normalize_mol(i, row, smiles_column='smiles'):

#     mol = dm.to_mol(str(row[smiles_column]), ordered=True)
#     mol = dm.fix_mol(mol)
#     mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
#     mol = dm.standardize_mol(mol, disconnect_metals=False, normalize=True, reionize=True, uncharge=False, stereo=True)

#     fingerprint_function = rdMolDescriptors.GetMorganFingerprintAsBitVect
#     pars = {
#         "radius": 2,
#         "nBits": 8192,
#         "invariants": [],
#         "fromAtoms": [],
#         "useChirality": False,
#         "useBondTypes": True,
#         "useFeatures": False,
#     }
#     fp = fingerprint_function(mol, **pars)

#     row["standard_smiles"] = dm.standardize_smiles(dm.to_smiles(mol))
#     row["selfies"] = dm.to_selfies(mol)
#     row["inchi"] = dm.to_inchi(mol)
#     row["inchikey"] = dm.to_inchikey(mol)
#     row["onbits_fp"] = list(fp.GetOnBits())

#     return row


# from pyarrow import fs
# import pyarrow.dataset as ds
# from tqdm.contrib import tenumerate
# from tqdm import tqdm
# from datetime import datetime
# s3  = fs.S3FileSystem(region="us-east-1")
# dataset = ds.dataset("molbeam/tested", filesystem=s3)
# now = datetime.now()
# fragments = [file for file in dataset.get_fragments()]
# total_frags = len(fragments)
# for count,record_batch in tenumerate(dataset.to_batches(columns=["canonical_ID", "enumerated_smiles", "achiral_fp"]), start=0, total=total_frags):
#     now = datetime.now()
#     current_time = now.strftime("%H:%M:%S")
#     print("Current Time =", current_time)

