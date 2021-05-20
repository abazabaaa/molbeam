import pyarrow.dataset as ds
from pyarrow import Table
from pyarrow import csv
import pyarrow as pa
from pyarrow.parquet import ParquetWriter
import pathlib
import pandas as pd
import pyarrow.feather as fe
import datamol as dm
import operator

dm.disable_rdkit_log()

dataset = ['/data/dockop_glide_d3/second50k_glide_molchunkout/second50k_glide_out.molchunk', '/data/dockop_glide_d3/first50k_glide_molchunkout', '/data/dockop_glide_d3/thirdd50k_glide_molchunkout/third50k_glide_out.molchunk', '/data/dockop_glide_d3/fourth50k_glide_molchunkout/fourth50k_glide_out.molchunk', '/data/dockop_glide_d3/fithround_glide_molchunkout/fifth50k_glide_out.molchunk']
dflist = []
for data in dataset:
    dataset = ds.dataset(data, format="feather")
    df = dataset.to_table().to_pandas()
    dflist.append(df)
    
def combine_unique_molchunks_with_identical_columns(molchunk_1, molchunk_2):
    outer_merged = pd.merge(molchunk_1, molchunk_2, how='outer')
    return outer_merged
  
docked_df = combine_unique_molchunks_with_identical_columns(dflist[0], dflist[1])
docked_df = combine_unique_molchunks_with_identical_columns(docked_df, dflist[2])
docked_df = combine_unique_molchunks_with_identical_columns(docked_df, dflist[3])
docked_df = combine_unique_molchunks_with_identical_columns(docked_df, dflist[4])

print(f'The mean docking score was {docked_df.docking_score.mean()}')
print(f'The median docking score was {docked_df.docking_score.median()}')
zero_one_percentile = docked_df.docking_score.quantile(0.001)
print(f'The 0.1th percentile was {docked_df.docking_score.quantile(0.001)}')
zero_three_percentile = docked_df.docking_score.quantile(0.003)
print(f'The 0.3th percentile was {docked_df.docking_score.quantile(0.003)}')
zero_five_percentile = docked_df.docking_score.quantile(0.005)
print(f'The 0.5th percentile was {docked_df.docking_score.quantile(0.005)}')
first_percentile = docked_df.docking_score.quantile(0.01)
print(f'The 1st percentile was {docked_df.docking_score.quantile(0.01)}')
fifth_percentile = docked_df.docking_score.quantile(0.05)
print(f'The 5th percentile was {docked_df.docking_score.quantile(0.05)}')
tenth_percentile = docked_df.docking_score.quantile(0.1)
print(f'The 10th percentile was {docked_df.docking_score.quantile(0.1)}')
print(f'The 25th percentile was {docked_df.docking_score.quantile(0.25)}')


docked_df_zero_three_percentile = docked_df[docked_df.docking_score < zero_three_percentile]
print(f'The number of ligands in the top 0.3 percentile is: {len(docked_df_zero_three_percentile)}')
docked_df_zero_five_percentile = docked_df[docked_df.docking_score < zero_five_percentile]
print(f'The number of ligands in the top 0.5 percentile is: {len(docked_df_zero_five_percentile)}')
docked_df_first_percentile = docked_df[docked_df.docking_score < first_percentile]
print(f'The number of ligands in the top 1 percentile is: {len(docked_df_first_percentile)}')
