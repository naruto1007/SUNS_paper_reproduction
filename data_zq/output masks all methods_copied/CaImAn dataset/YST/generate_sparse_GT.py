# generate GT sparse mat for subsequent performance cal
# mat with _sparse, _sparse_original, _sparse_generated is the same in YST/GT Masks

import os
import glob
import h5py
import numpy as np
from scipy import sparse
from scipy.io import savemat, loadmat


def mat_generate_sparse_GT_original(dir_Masks):
    dir_all = glob.glob(os.path.join(dir_Masks, '*FinalMasks*.mat'))
    for path_name in dir_all:
        file_name = os.path.split(path_name)[1]
        if '_sparse' not in file_name:
            print(file_name)
            try:  # If file_name is saved in '-v7.3' format
                mat = h5py.File(path_name, 'r')
                FinalMasks = np.array(mat['FinalMasks']).astype('bool')
                mat.close()
            except OSError:  # If file_name is not saved in '-v7.3' format
                mat = loadmat(path_name)
                FinalMasks = np.array(mat["FinalMasks"]).transpose([2, 1, 0]).astype('bool')

            (ncells, Ly, Lx) = FinalMasks.shape
            GTMasks_2 = sparse.coo_matrix(FinalMasks.reshape(ncells, Lx * Ly).T)
            savemat(os.path.join(path_name[:-4] + '_sparse_original.mat'), \
                    {'GTMasks_2': GTMasks_2}, do_compression=True)


def mat_generate_sparse_GT(dir_Masks):
    dir_all = glob.glob(os.path.join(dir_Masks, '*FinalMasks*.mat'))
    for path_name in dir_all:
        file_name = os.path.split(path_name)[1]
        if '_sparse' not in file_name:
            print(file_name)
            try:  # If file_name is saved in '-v7.3' format
                mat = h5py.File(path_name, 'r')
                FinalMasks = np.array(mat['FinalMasks']).transpose([1, 2, 0]).astype('bool')
                mat.close()
            except OSError:  # If file_name is not saved in '-v7.3' format
                mat = loadmat(path_name)
                FinalMasks = np.array(mat["FinalMasks"]).transpose([1, 2, 0]).astype('bool')

            (Lx, Ly, ncells) = FinalMasks.shape
            GTMasks_2 = sparse.coo_matrix(FinalMasks.reshape(Lx * Ly, ncells))
            savemat(os.path.join(path_name[:-4] + '_sparse_generated.mat'), \
                    {'GTMasks_2': GTMasks_2}, do_compression=True)


# the results from the two GT_methods is the same
dir_Masks_GT = "D:\PyCharm_project\SUNS_paper_reproduction\data_zq\CalmAn_dataset\YST\GT Masks"
mat_generate_sparse_GT(dir_Masks_GT)
