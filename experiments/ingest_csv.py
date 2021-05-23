
import resource
from time import sleep

from pyarrow import Table
from pyarrow import csv
import pyarrow as pa
from pyarrow.parquet import ParquetWriter

import pandas as pd

import click

from memory_profiler import profile

from datetime import timedelta
from timeit import time
import datamol as dm
import pyarrow as pa
import pandas as pd
import numpy as np

# def main(file, output_dir, output_prefix, column1, column2):


def stopwatch(method):
    def timed(*args, **kw):
        ts = time.perf_counter()
        result = method(*args, **kw)
        te = time.perf_counter()
        duration = timedelta(seconds=te - ts)
        print(f"{method.__name__}: {duration}")
        return result
    return timed


class InputStreamReader:
    def __init__(self, file_stream):
        self.file_stream = file_stream
        self._stream = None

    def batches(self):
        i = tries = 0
        while True:
            try:
                batch = self.__next_batch()
                i += 1
                yield i, batch
            except StopIteration:
                break

    def __next_batch(self):
        return self.stream.read_next_batch()


    @property
    def stream(self):
        if not self._stream:
            read_options = pa.csv.ReadOptions(block_size=1048576000)
            parse_options = pa.csv.ParseOptions(delimiter='\t')
            convert_options = pa.csv.ConvertOptions(include_columns=include_columns)
            self._stream = pa.csv.open_csv(
                self.file_stream, read_options=read_options,
                parse_options=parse_options,
                convert_options=convert_options

        )
        return self._stream


@stopwatch        
def csv_stream_to_parquet_batch_writer(include_columns, \
                 input_file_to_stream, \
                 output_file_stream_directory, \
                 output_file_prefix):
        
        print('Initiating stream.')
        
        input_stream_reader = InputStreamReader(input_file_to_stream)
        
        outfiles_list = []
        
        for i, batch in input_stream_reader.batches():
            print(f'Ingesting batch number {i}')
            df = batch.to_pandas()
            table = pa.Table.from_pandas(df)
            schema = table.schema
#             smiles = list(df[smiles_column_title])
#             print(f'Writing a total of {len(smiles)} smiles per output file to disk.')
            outfile = f'{output_file_stream_directory}{output_file_prefix}_{i}.parquet'
            ParquetWriter(outfile, schema).write_table(table)
            print(f'Wrote parquet to {outfile}')
            outfiles_list.append(outfile)
            
        return outfile



file = '/data/dopamine_3_results/d3_glide_dockop/random_50k_smiles_try2.smi'
output_dir = '/data/dopamine_3_results/d3_glide_dockop/first_random50k_glide/'
output_prefix = '50k_glide_output_smiles'
include_columns = ['Title', 'docking score']
column1 = 'SMILES'

outfiles = csv_stream_to_parquet_batch_writer(include_columns, \
                     file, \
                     output_dir, \
                     output_prefix)

print(outfiles)
