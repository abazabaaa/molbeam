import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.flight as fl

import pandas as pd
import dataframe_image as dfi
from rdkit.Chem import PandasTools
from tqdm import tqdm

import molbeam
import similarity


def process_batch(query_names, query_matrix, mol_batch, threshold=.99):
    fingerprints = mol_batch.column('achiral_fp')
    fp_matrix_in = similarity.build_fingerprint_matrix(fingerprints)
    fp_distance = similarity.fast_jaccard(fp_matrix_in, query_matrix)

    # if elements are not below threshold don't keep them
    matches = fp_distance <= threshold
    result = None
    if matches.any():
        result = pd.DataFrame(fp_distance, columns=['PBB3', 'PM_PBB3', 'C5_05'])
        result.insert(0, 'canonical_id', mol_batch.column('canonical_ID'))
        result.insert(1, 'std_smiles', mol_batch.column('enumerated_smiles'))

    return result


def export_results(result_list, threshold):
    result_df = pd.concat(result_list)
    table = pa.Table.from_pandas(result_df)
    pq.write_table(table, 'query_out.parquet')
    print('Wrote results to: query_out.parquet')

    con = duckdb.connect(database=':memory:', read_only=False)
    sql = '''
        SELECT
            *
        FROM parquet_scan('query_out.parquet')
        WHERE PBB3 < ?
    '''
    top_results = con.execute(sql, [threshold]).fetchdf()

    PandasTools.AddMoleculeColumnToFrame(top_results, smilesCol='std_smiles', molCol='self')
    final_df = top_results.sort_values(by=['PBB3'], ascending=True)
    final_df = final_df.reset_index(drop=True)

    file_name = 'search_results.png'
    dfi.export(final_df, file_name)


def main():
    query = [
        ('PBB3', 'OC1=CC=C(N=C(/C=C/C=C/C2=CN=C(NC)C=C2)S3)C3=C1'),
        ('PM_PBB3', 'OC(CF)COC1=CC=C(N=C(/C=C/C=C/C2=CN=C(NC)C=C2)S3)C3=C1'),
        ('C5_05', 'OC(CF)COC1=CC=C(N=C(/C=C/C#CC2=CN=C(NC)C=C2)S3)C3=C1'),
    ]
    query_names = [q[0] for q in query]
    query_matrix = similarity.format_query(query)
    columns = ["canonical_ID", "enumerated_smiles", "achiral_fp"]
    results = []
    print('Searching enamine database of 3.8M molecules...')

    # minimum jaccard distance to be considered a match
    threshold = 0.75
    for mol_batch in molbeam.stream(batch_size=20000, columns=columns):
        continue
    #     result = process_batch(query_names, query_matrix, mol_batch, threshold)
    #     if result is not None:
    #         results.append(result)

    # export_results(results, threshold)


def client():
    # client = fl.connect("grpc://0.0.0.0:8815")
    client = fl.connect("grpc://35.168.111.94:8815")

    stream = client.do_get(fl.Ticket('molbeam'))
    for r in tqdm(stream, total=191):
        continue
        # print(r.data.to_pandas())


def pclient():
    import ray
    ray.init()

    @ray.remote
    def f(batch):
        return 1

    client = fl.connect("grpc://35.168.111.94:8815")
    stream = client.do_get(fl.Ticket('molbeam'))
    futures = [f.remote(b.data) for b in tqdm(stream, total=191)]
    print(ray.get(futures))


if __name__ == '__main__':
    main()
