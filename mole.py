import datamol as dm
from rdkit.Chem import rdMolDescriptors
# from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit import DataStructs
# from rdkit.Chem import rdMolDescriptors

dm.disable_rdkit_log()


def normalize_mol(i, row, smiles_column='smiles'):

    mol = dm.to_mol(str(row[smiles_column]), ordered=True)
    mol = dm.fix_mol(mol)
    mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
    mol = dm.standardize_mol(mol, disconnect_metals=False, normalize=True, reionize=True, uncharge=False, stereo=True)

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

    row["standard_smiles"] = dm.standardize_smiles(dm.to_smiles(mol))
    row["selfies"] = dm.to_selfies(mol)
    row["inchi"] = dm.to_inchi(mol)
    row["inchikey"] = dm.to_inchikey(mol)
    row["onbits_fp"] = list(fp.GetOnBits())

    return row
