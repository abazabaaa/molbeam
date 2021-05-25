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
        fingerprint_matrix_in = similarity.build_fingerprint_matrix(mol_batch)
        fingerprint_diff = similarity.fast_jaccard(fingerprint_matrix_in, query_matrix)
        print(fingerprint_diff)


if __name__ == '__main__':
    main()