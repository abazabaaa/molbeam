import pandas as pd
import numpy as np
import molbeam
import similarity


def main():
    query = [
        ('PBB3', 'OC1=CC=C(N=C(/C=C/C=C/C2=CN=C(NC)C=C2)S3)C3=C1'),
        ('PM_PBB3', 'OC(CF)COC1=CC=C(N=C(/C=C/C=C/C2=CN=C(NC)C=C2)S3)C3=C1'),
        ('C5_05', 'OC(CF)COC1=CC=C(N=C(/C=C/C#CC2=CN=C(NC)C=C2)S3)C3=C1'),
    ]
    query_matrix = similarity.format_query(query)
    columns = ["canonical_ID", "enumerated_smiles", "achiral_fp"]


    for mol_batch in molbeam.stream(batch_size=20000, columns=columns):



        fingerprints = mol_batch.column('achiral_fp')
        fp_matrix_in = similarity.build_fingerprint_matrix(fingerprints)
        fp_distance = similarity.fast_jaccard(fp_matrix_in, query_matrix)
        # if elements are below threshold save results
        # else keep going



        df = pd.DataFrame(fp_distance, columns=['PBB3', 'PM_PBB3', 'C5_05'])
        df.insert(0, 'canonical_id', mol_batch.column('canonical_ID'))
        df.insert(1, 'std_smiles', mol_batch.column('enumerated_smiles'))
        print(df.head(5))
        break


if __name__ == '__main__':
    main()