import sys
import config
from rich import print
import pathlib
from rich.console import Console


console = Console()

dataset_dir = sys.argv[1]
output_dir = sys.argv[2]
out_prefex = sys.argv[3]

# dataset_dir = '/data/mol_chunk_tests_cluster/parquet'
# output_dir = '/data/mol_chunk_tests_cluster/out_dir'
# out_prefix = 'testing'

# Load the dataset from parquet one by one
dataset = ds.dataset(dataset_dir, format=config.format)

# Create a list of fragments that are not memory loaded
fragments = [file for file in dataset.get_fragments()]

total_frags = len(fragments)
print(f'There are a total of {len(fragments)} fragments in the dataset: {output_dir}')

n_jobs = config.n_jobs
smiles_col = config.smiles_col
catalog_id_col = config.catalog_id_col
canonical_id_col = config.canonical_id_col


for count,element in tenumerate(fragments, start=0, total=None):
    

    console.print(f'Starting fragment dataset: {count} in {dataset_dir}', style="blue bold")
    #cast the fragment as a pandas df
    df = element.to_table().to_pandas()
    df2 = prep_parquet_db(df, n_jobs, smiles_col, catalog_id_col, canonical_id_col)

    console.print(f'There are a total of {len(df2)} valid smiles strings after stereoisomer enumeration', style="green bold")

    table = pa.Table.from_pandas(df2, preserve_index=False)

    
    #write the molchunk to disk
    pq.write_table(table, f'{output_dir}/{out_prefix}_{count}.molchunk')
    console.print(f'Wrote parquet to {output_dir}/{out_prefix}_{count}.molchunk', style="purple bold")
