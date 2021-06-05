from datetime import timedelta
from timeit import time

import pyarrow as pa

import numpy as np
import numpy.ma as ma

from scipy import sparse

import mole


def stopwatch(method):
    def timed(*args, **kw):
        ts = time.perf_counter()
        result = method(*args, **kw)
        te = time.perf_counter()
        duration = timedelta(seconds=te - ts)
        print(f"{method.__name__}: {duration}")
        return result
    return timed


# @stopwatch
# def fingerprint_matrix_from_df(fp_list_query):
#     # smiles = list(df['enumerated_smiles'])
#     onbits_fp = fp_list_query
#     # zincid = list(df['canonical_ID'])

#     # name_list = []
#     row_idx = list()
#     col_idx = list()
#     # num_on_bits = []
#     for count, m in enumerate(smiles):
#         # zincid_name = str(zincid[count])
#         onbits = list(onbits_fp[count])

#         col_idx += onbits
#         row_idx += [count] * len(onbits)
#         num_bits = len(onbits)
#         # num_on_bits.append(num_bits)
#         # name_list.append(zincid_name)

#     unfolded_size = 8192
#     fingerprint_matrix = sparse.coo_matrix((np.ones(len(row_idx)).astype(bool), (row_idx, col_idx)),
#               shape=(max(row_idx)+1, unfolded_size))
#     fingerprint_matrix =  sparse.csr_matrix(fingerprint_matrix)
#     fp_mat = fingerprint_matrix
#     return fp_mat


def fast_jaccard(X, Y=None):
    """credit: https://stackoverflow.com/questions/32805916/compute-jaccard-distances-on-sparse-matrix"""
    if isinstance(X, np.ndarray):
        X = sparse.csr_matrix(X)
    if Y is None:
        Y = X
    else:
        if isinstance(Y, np.ndarray):
            Y = sparse.csr_matrix(Y)
    assert X.shape[1] == Y.shape[1]

    X = X.astype(bool).astype(int)
    Y = Y.astype(bool).astype(int)
    intersect = X.dot(Y.T)
    x_sum = X.sum(axis=1).A1
    y_sum = Y.sum(axis=1).A1
    xx, yy = np.meshgrid(x_sum, y_sum)
    union = ((xx + yy).T - intersect)
    return (1 - intersect / union).A


# def min_distance(row):

#     a = out_array[rowIndex(row)]
#     minval = np.min(ma.masked_where(a==0, a))
#     minvalpos = np.argmin(ma.masked_where(a==0, a))
# #     sumval = np.sum(ma.masked_where(a==0, a))

#     smiles_nn = smiles[minvalpos]
#     name_nn = name[minvalpos]

#     return smiles_nn, name_nn, minval


# def rowIndex(row):
#     return row.name


def table_to_csr_fp(table):
    x = table.column('achiral_fp')
    row_idx = list()
    col_idx = list()
    num_on_bits = []
    for _ in range(len(x)):
        onbits = pa.ListValue.as_py(x[_])
        col_idx += onbits
        # print(f'col_idx: {col_idx}')
        row_idx += [_] * len(onbits)
        num_bits = len(onbits)
        num_on_bits.append(num_bits)

    unfolded_size = 8192
    fingerprint_matrix = sparse.coo_matrix((np.ones(len(row_idx)).astype(bool), (row_idx, col_idx)), shape=(max(row_idx)+1, unfolded_size))
    fingerprint_matrix =  sparse.csr_matrix(fingerprint_matrix)
    return fingerprint_matrix


def build_fingerprint_matrix(fingerprints):
    col_idx = fingerprints.flatten().to_numpy()
    row_idx = fingerprints.value_parent_indices().to_numpy()
    unfolded_size = 8192
    fingerprint_matrix = sparse.coo_matrix((np.ones(len(row_idx)).astype(bool), (row_idx, col_idx)),
              shape=(max(row_idx)+1, unfolded_size))
    fingerprint_matrix =  sparse.csr_matrix(fingerprint_matrix)
    return fingerprint_matrix


def format_query(query):
    smiles_list = []
    fp_list = []
    for q in query:
        # query_name = q[0]
        smiles = q[1]
        std_smiles, fingerprint = mole.smiles_to_fingerprint(smiles)
        smiles_list.append(std_smiles)
        fp_list.append(fingerprint)

    table = pa.table(
        [smiles_list, fp_list],
        names=['std_smiles', 'achiral_fp'],
    )
    # fp = build_fingerprint_matrix(table)
    query_fp_matrix = table_to_csr_fp(table)
    return query_fp_matrix
