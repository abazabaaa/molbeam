import sys
from set_up import Setup
from estimator import CommonEstimator
import json
import h5py
import glob
import os
from scipy import sparse
import numpy as np
#from utils import get_memory_usage
import pyarrow as pa
import numpy as np
import pyarrow.feather as feather
import pandas as pd
SEED = 12939 #from random.org
np.random.seed(SEED)


import itertools


def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk
# def write_results(preds, fpsize, trainingSize, name, repeat_number):
#         """Writes an HDF5 file that stores the results. 
#         preds: np.array: prediction scores for the test samples
#         fpsize: int: size the fingerprint was folded to
#         name: str: the estimator name, as stored in the json
#         repeat_number: int.
 
#         Results stored are:
#         - test indices
#         - preds 
#         and there should be one set of results for each repeat."""

#         #write the first time, append afterwards. 
#         write_option = 'w' if repeat_number==0 else 'a'
#         outf = h5py.File('../processed_data/'+self.fingerprint_kind+'_'+str(fpsize)+'_'+str(trainingSize)+'_'+name+'.hdf5', write_option)

#         rp = outf.create_group(f'repeat{repeat_number}')

#         dset_idx = rp.create_dataset('test_idx', self.test_idx.shape, dtype='int')
#         dset_idx[:] = self.test_idx

#         dset_pred = rp.create_dataset('prediction', preds.shape, dtype='float16')
#         dset_pred[:] = preds
        
#         outf.close()

def fold_fingerprints(feature_matrix):
    """Folds a fingerprint matrix by bitwise OR.
    (scipy will perform the bitwise OR because the `data` is bool,
    and it will not cast it to int when two Trues are added."""

    ncols = feature_matrix.shape[1]
    return feature_matrix[:,:ncols//2] + feature_matrix[:,ncols//2:]

def fold_to_size(size, fingerprints):
    """Performs the `fold` operation multiple times to reduce fp 
    length to the desired size."""
    feature_matrix = fingerprints
    while feature_matrix.shape[1]>size:
        feature_matrix = fold_fingerprints(feature_matrix)
    return feature_matrix

def random_split(self, number_train_ligs):
    """Simply selects some test and train indices"""



fpType = sys.argv[1]
fpSize = int(sys.argv[2])
# trainingSetSize = int(sys.argv[3])
json_name = sys.argv[3]
# dataset = sys.argv[5]



print('Running:')
print(f'python main.py {fpType} {fpSize} {json_name}')


estimators = json.load(open(json_name, 'r'))['estimators']

if __name__=='__main__':
    #setup the data:
    setup = Setup(fpType, verbose=True)
    # try:
    #     setup.write_fingerprints()
    # except:
    #     print('Already written fpfile')        
    # setup.load_fingerprints()
    # setup.load_scores()
    
    
# +input_db_ext)
    
    dataset_train = '/data/dockop_glide_d3/fourth50k_glide_fp/processed_data'

    fingerprint_file_ext = ".npz"
    scores_file_ext = ".feather"
    fingerprint_file_names_list_train = glob.glob(os.path.join(dataset_train+"*"+fingerprint_file_ext))

    fingerprint_files_list_train = [(dataset_train+'{:01d}'.format(x)+ fingerprint_file_ext) for x in range(len(fingerprint_file_names_list_train))]
    scores_files_list_train = [(dataset_train+'{:01d}'.format(y)+ scores_file_ext) for y in range(len(fingerprint_file_names_list_train))]

    npz_list_train = []
    scores_list_train = []
    names_list_train = []
    smiles_list_train = []
    for count, batch in enumerate(fingerprint_file_names_list_train):
        fingerprints = sparse.load_npz(fingerprint_files_list_train[count])
        df = feather.read_feather(scores_files_list_train[count])
        scores = list(df['scores'])
        smiles = list(df['smiles'])
        names = list(df['names'])
        npz_list_train.append(fingerprints)
        scores_list_train.append(scores)
        names_list_train.append(names)
        smiles_list_train.append(smiles)

    flat_sparse_fingerprints_train = sparse.vstack(npz_list_train)

    flat_scores_list_train = [item for sublist in scores_list_train for item in sublist]
    flat_names_list_train = [item for sublist in names_list_train for item in sublist]
    flat_smiles_list_train = [item for sublist in smiles_list_train for item in sublist]
    np_scores= np.array(flat_scores_list_train, dtype=np.float16)
    # np_scores = np.concatenate(scores_arry)
    num_ligs_train = len(flat_scores_list_train)
    print(f'the total number of batches within the training set is: {len(fingerprint_file_names_list_train)}')
    print(f'the number of ligands within the training set is : {num_ligs_train}')

    feature_matrix_train = fold_to_size(fpSize, flat_sparse_fingerprints_train)
    print(feature_matrix_train.shape)

    dataset_test = '/data/dopamine_3_results/150M_fps/processed_data'
    fingerprint_file_names_list_test = glob.glob(os.path.join(dataset_test+"*"+fingerprint_file_ext))
    fingerprint_files_list_test = [(dataset_test+'{:01d}'.format(x)+ fingerprint_file_ext) for x in range(len(fingerprint_file_names_list_test))]
    scores_files_list_test = [(dataset_test+'{:01d}'.format(y)+ scores_file_ext) for y in range(len(fingerprint_file_names_list_test))]
    num_batches = 80

    for count,c in enumerate(chunked_iterable(range(5973), size=175)):
        print(f'beginning batch number {count}')

        npz_list_test = []

        names_list_test = []
        smiles_list_test = []
        for batch_num in range(c[0], c[-1]):
            fingerprints = sparse.load_npz(fingerprint_files_list_test[batch_num])
            df = feather.read_feather(scores_files_list_test[batch_num])
            smiles = list(df['smiles'])
            names = list(df['names'])
            npz_list_test.append(fingerprints)

            names_list_test.append(names)
            smiles_list_test.append(smiles)

        flat_sparse_fingerprints_test = sparse.vstack(npz_list_test)


        flat_names_list_test = [item for sublist in names_list_test for item in sublist]
        flat_smiles_list_test = [item for sublist in smiles_list_test for item in sublist]

        # np_scores = np.concatenate(scores_arry)
        num_ligs_test = len(flat_names_list_test)
        print(f'the total number of batches within the test set is: {len(fingerprint_files_list_test)}')
        print(f'the  number of batches selected for use with the model is: {num_batches}')
        print(f'the number of ligands within the test set is : {num_ligs_test}')
        feature_matrix_test = fold_to_size(fpSize, flat_sparse_fingerprints_test)
        print(feature_matrix_test.shape)
        # #evaluation stuff goes here:    
        for estimator in estimators:

            for repeat in range(1):
                idx_train = np.arange(num_ligs_train)
                idx_test = np.arange(num_ligs_test)

                np.random.shuffle(idx_train)
                np.random.shuffle(idx_test)
    ##### Deal with id.
                # train_idx = idx[:trainingSetSize]
                # test_idx = idx[trainingSetSize:]
                train_idx = idx_train
                test_idx = idx_test

                # training_smi = [flat_smiles_list[i] for i in train_idx]
                test_smi = [flat_smiles_list_test[i] for i in test_idx]


                # training_names = [flat_names_list[i] for i in train_idx]
                test_names = [flat_names_list_test[i] for i in test_idx]

                # training_scores = [flat_scores_list[i] for i in train_idx]
                # test_scores = [flat_scores_list[i] for i in test_idx]
                
                common_estimator = CommonEstimator(estimator, cutoff=0.3, verbose=True)
                print(train_idx.shape)
                print(np_scores.shape)
                common_estimator.fit(feature_matrix_train, np_scores)
                pred = common_estimator.chunked_predict(feature_matrix_test)
                pred_list = [pred[i] for i in range(len(pred))]
                print(f'length of prediction list is {len(pred_list)}')
                print(f'length of smiles is {len(test_smi)}')
                print(f'length of names is {len(test_names)}')
                # print(f'length of scores is {len(test_scores)}')

        #         # scores = [scores_list[i] for i in range(len(scores_list))]

                pred_list_pa = pa.array(pred_list)   
                smiles_pa = pa.array(test_smi, type=pa.string())
                # scores_pa = pa.array(test_scores)
                names_pa = pa.array(test_names, type=pa.string())

                data = [
                pred_list_pa,
                smiles_pa,
                names_pa
                ]

                batch_from_data = pa.RecordBatch.from_arrays(data, ['pred_list', 'smiles', 'names'])
                df = batch_from_data.to_pandas()
                feather.write_feather(df, f'/data/dockop_glide_d3/fourth50k_predout/fourth50k_{count}_set{repeat}.feather')

            # setup.write_results(pred, fpSize, trainingSetSize, estimator['name'], repeat, test_idx)

        print(f'completed batch number {count}')
        # idx_pre_shuffled_pa = pa.array(idx_list_pre_shuffle, type=pa.int64())        
        # idx_shuffled_pa = pa.array(idx_list_shuffled, type=pa.int64())

        # data = [
        # pred_list_pa,
        # idx_pre_shuffled_pa,
        # idx_shuffled_pa,
        # smiles_pa,
        # scores_pa,
        # names_pa
        # ]
