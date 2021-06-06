import pyarrow.dataset as ds
from tqdm import tqdm


def stream(batch_size=20000, columns=["canonical_ID", "enumerated_smiles", "achiral_fp"]):
    dataset = ds.dataset("s3://molbeam/tested", format="parquet")
    num_files = len(dataset.files)
    print(num_files)
    return tqdm(dataset.to_batches(columns=["canonical_ID", "enumerated_smiles", "achiral_fp"]), total=num_files)

