# import pyarrow.dataset as ds
# from tqdm import tqdm


# def stream(batch_size=20000, columns=["canonical_ID", "enumerated_smiles", "achiral_fp"]):
#     dataset = ds.dataset("s3://molbeam/tested", format="parquet")
#     num_files = len(dataset.files)
#     return tqdm(dataset.to_batches(columns=["canonical_ID", "enumerated_smiles", "achiral_fp"]), total=num_files)

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.flight as flight
import numpy as np
import pandas as pd
import time
import glob
import threading
from tqdm.contrib import tenumerate
import boto3
import os
import pyarrow
import urllib.parse

from pyarrow import fs
import s3fs

from pyarrow.util import find_free_port

from datetime import timedelta
from timeit import time

def stopwatch(method):
    def timed(*args, **kw):
        ts = time.perf_counter()
        result = method(*args, **kw)
        te = time.perf_counter()
        duration = timedelta(seconds=te - ts)
        print(f"{method.__name__}: {duration}")
        return result
    return timed

class Driver:
    _s3_client = None
    _s3_resource = None

    @staticmethod
    def s3_client():
        if Driver._s3_client is None:
            Driver._s3_client = boto3.client('s3')
#             print("returning Driver._s3_client")
        return Driver._s3_client

    @staticmethod
    def s3_resource():
        if Driver._s3_resource is None:
            Driver._s3_resource = boto3.resource('s3')
#             print("return Driver._s3_resource")
        return Driver._s3_resource

    @staticmethod
    def list(url):
        parts = urllib.parse.urlparse(url)

        # S3
        if parts.scheme == 's3':
            bucket = parts.netloc
#             print(bucket)
            key = parts.path[1:] + '/'
#             print(key)
            pages = Driver.s3_client().get_paginator(
                'list_objects_v2').paginate(Bucket=bucket, Prefix=key)
            for page in pages:
                if 'Contents' in page.keys():
                    for obj in page['Contents']:
#                         print(obj)
                        yield 's3://{}/{}'.format(bucket, obj['Key'])

        # File System
        elif parts.scheme == 'file':
            path = os.path.join(parts.netloc, parts.path)
            for fn in os.listdir(path):
                if os.path.isfile(os.path.join(path, fn)):
                    yield 'file://' + os.path.join(path, fn)

        else:
            raise Exception('URL {} not supported'.format(url))

    @staticmethod
    def read_metadata(url):
        parts = urllib.parse.urlparse(url)

        # S3
        if parts.scheme == 's3':
            bucket = parts.netloc
            key = parts.path[1:] + '/metadata'
            obj = Driver.s3_client().get_object(Bucket=bucket, Key=key)
            return obj['Body'].read().decode('utf-8')

        # File System
        elif parts.scheme == 'file':
            path = os.path.join(parts.netloc, parts.path, 'metadata')
            return open(path).read()

        else:
            raise Exception('URL {} not supported'.format(url))

    @staticmethod
    def create_reader(url, compression=None):
        parts = urllib.parse.urlparse(url)

        # S3
        if parts.scheme == 's3':
            bucket = parts.netloc
            print(bucket)
            key = parts.path[1:]
            print(key)
            obj = Driver.s3_client().get_object(Bucket=bucket, Key=key)
#             print(obj)
            buf = obj['Body'].read()
#             print(buf)
            strm = pyarrow.input_stream(pyarrow.py_buffer(buf),
                                        compression=compression)
            return pyarrow.RecordBatchStreamReader(strm)

        # File System
        elif parts.scheme == 'file':
            path = os.path.join(parts.netloc, parts.path)
            strm = pyarrow.input_stream(path, compression=compression)
            return pyarrow.RecordBatchStreamReader(strm)

        else:
            raise Exception('URL {} not supported'.format(url))

    @staticmethod
    def create_writer(url, schema, compression=None):
        parts = urllib.parse.urlparse(url)

        # S3
        if parts.scheme == 's3':
            bucket = parts.netloc
            key = parts.path[1:]
            buf = pyarrow.BufferOutputStream()
            stream = pyarrow.output_stream(buf, compression=compression)
            writer = pyarrow.RecordBatchStreamWriter(stream, schema)

            try:
                yield writer
            except GeneratorExit:
                writer.close()
                stream.close()
                Driver.s3_client().put_object(Body=buf.getvalue().to_pybytes(),
                                              Bucket=bucket,
                                              Key=key)

        # File System
        elif parts.scheme == 'file':
            path = os.path.join(parts.netloc, parts.path)
            stream = pyarrow.output_stream(path, compression=compression)
            writer = pyarrow.ipc.RecordBatchStreamWriter(stream, schema)

            try:
                yield writer
            except GeneratorExit:
                writer.close()
                stream.close()

        else:
            raise Exception('URL {} not supported'.format(url))

    @staticmethod
    def delete_all(url):
        parts = urllib.parse.urlparse(url)

        # S3
        if parts.scheme == 's3':
            bucket = parts.netloc
            key = parts.path[1:]
            Driver.s3_resource().Bucket(
                bucket).objects.filter(Prefix=key).delete()

        # File System
        elif parts.scheme == 'file':
            path = os.path.join(parts.netloc, parts.path)
            for fn in os.listdir(path):
                os.unlink(os.path.join(path, fn))

        else:
            raise Exception('URL {} not supported'.format(url))

    @staticmethod
    def delete(url):
        parts = urllib.parse.urlparse(url)

        # S3
        if parts.scheme == 's3':
            bucket = parts.netloc
            key = parts.path[1:]
            Driver.s3_client().delete_object(Bucket=bucket, Key=key)

        # File System
        elif parts.scheme == 'file':
            path = os.path.join(parts.netloc, parts.path)
            os.unlink(path)

        else:
            raise Exception('URL {} not supported'.format(url))
            
def get_s3_dataset(url):

    x = Driver.list(url)
    y = list(x)

    url = "s3://molbeam/tested"
    lis_p = []
    for count,crt_url in enumerate(y):
    #     print(crt_url)
#         print(count)
        if y[count] == '/'.join((url, '')):
#             print("passing")
            print(y[count])
            pass
        else:
#             print('found_something')
#             print(y[count])
            lis_p.append(y[count])
    
    filesystem=s3fs.S3FileSystem()
    # table = dataset.read()


    dataset = ds.dataset(lis_p, format="parquet", filesystem=filesystem)
    return dataset, lis_p

class DemoServer(flight.FlightServerBase):
    
    def __init__(self, location):
        self._cache = {}
        super().__init__(location)
    
    def list_actions(self, context):
        return [flight.ActionType('list-tables', 'List stored tables'),
                flight.ActionType('drop-table', 'Drop a stored table')]

    # -----------------------------------------------------------------
    # Implement actions
    
    def do_action(self, context, action):
        handlers = {
            'list-tables': self._list_tables,
            'drop-table': self._drop_table
        }        
        handler = handlers.get(action.type)
        if not handler:
            raise NotImplementedError   
        return handlers[action.type](action)
        
    def _drop_table(self, action):
        del self._cache[action.body]
        
    def _list_tables(self, action):
        return iter([flight.Result(cache_key) 
                     for cache_key in sorted(self._cache.keys())])

    # -----------------------------------------------------------------
    # Implement puts
    
    def do_put(self, context, descriptor, reader, writer):
        self._cache[descriptor.command] = reader.read_all()
        
    # -----------------------------------------------------------------
    # Implement gets

    def do_get(self, context, ticket):
        table = self._cache[ticket.ticket]
        return flight.RecordBatchStream(table)
    
    

port = 1337
location = flight.Location.for_grpc_tcp("localhost", find_free_port())
location

server = DemoServer(location)

thread = threading.Thread(target=lambda: server.serve(), daemon=True)
thread.start()

class DemoClient:
    
    def __init__(self, location, options=None):
        self.con = flight.connect(location)
        self.con.wait_for_available()
        self.options = options
        
    # Call "list-tables" RPC and return results as Python list
    def list_tables(self):
        action = flight.Action('list-tables', b'')
        return [x.body.to_pybytes().decode('utf8') for x in self.con.do_action(action)]    

    # Send a pyarrow.Table to the server to be cached
    def cache_table_in_server(self, name, table):
        desc = flight.FlightDescriptor.for_command(name.encode('utf8'))
        put_writer, put_meta_reader = self.con.do_put(desc, table.schema,
                                                      options=self.options)
        put_writer.write(table)
        put_writer.close()

    # Request a pyarrow.Table by name
    def get_table(self, name):
        reader = self.con.do_get(flight.Ticket(name.encode('utf8')),
                                 options=self.options)
        return reader.read_all()

    def list_actions(self):
        return self.con.list_actions()

ipc_options = pa.ipc.IpcWriteOptions(compression='zstd')
options = flight.FlightCallOptions(write_options=ipc_options)
client = DemoClient(location, options=options)

dataset, files_list = get_s3_dataset("s3://molbeam/tested")

for count, table in tenumerate(dataset.to_batches(columns=["canonical_ID", "enumerated_smiles", "achiral_fp"]), total=len(files_list)):

    client.cache_table_in_server(files_list[count], table)

@stopwatch
def get_single_table_from_flight_server(target):
    table_received = client.get_table(target)
    return table_received

recieved_table = get_table_from_flight_server(files_list[0])
print(recieved_table)
