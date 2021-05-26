import pandas as pd

import molbeam
import similarity


def process_batch(query_names, query_matrix, mol_batch, threshold=.75):
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

    print('Searching enamine database of 3,820,000 molecules...')
    for mol_batch in molbeam.stream(batch_size=20000, columns=columns):
        # minimum jaccard distance to be considered a match
        threshold = 0.75
        result = process_batch(query_names, query_matrix, mol_batch, threshold)
        if result is not None:
            results.append(result)

    final_df = pd.concat(results).sort_values(by=query_names)
    print(final_df.head(20))


if __name__ == '__main__':
    main()