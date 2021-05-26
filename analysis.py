import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit.Chem import PandasTools
import dataframe_image as dfi



def query_fp_search(df, distance):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, 'query_out.parquet')
    print('Wrote results to: query_out.parquet')
    con = duckdb.connect(database=':memory:', read_only=False)
    df2 = con.execute("SELECT * FROM parquet_scan('query_out.parquet') WHERE PBB3 <?", [distance]).fetchdf()
    print(df2.head(10))

    PandasTools.AddMoleculeColumnToFrame(df2, smilesCol='std_smiles', molCol='self')
    final_df = df2.sort_values(by=['PBB3'], ascending=True)
    final_df = final_df.reset_index(drop=True)
    print(len(final_df))
    results = 'df_styled.png'
    dfi.export(final_df, results)
    return results


