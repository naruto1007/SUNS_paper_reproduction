import os
import glob
import h5py
import numpy as np
from scipy import sparse
from scipy.io import savemat, loadmat


def mat_generate_sparse_output(dir_Masks):
    dir_all = glob.glob(os.path.join(dir_Masks, '*_neurons*.mat'))
    for path_name in dir_all:
        file_name = os.path.split(path_name)[1]
        if '_sparse' not in file_name:
            print(file_name)
            try:  # If file_name is saved in '-v7.3' format
                mat = h5py.File(path_name, 'r')
                Masks = np.array(mat['finalSegments']).transpose([1, 2, 0]).astype('bool')
                mat.close()
            except OSError:  # If file_name is not saved in '-v7.3' format
                mat = loadmat(path_name)
                Masks = np.array(mat["finalSegments"]).transpose([1, 2, 0]).astype('bool')

            (Lx, Ly, ncells) = Masks.shape
            Masks_2 = sparse.coo_matrix(Masks.reshape(Lx * Ly, ncells))
            savemat(os.path.join(path_name[:-4] + '_sparse.mat'), \
                    {'Masks': Masks_2}, do_compression=True)

# dir_Masks_output = "D:\PyCharm_project\SUNS_paper_reproduction\data_zq\output masks all methods_copied\CaImAn dataset\YST\SUNS noSF output_masks"
dir_Masks_output = "."
mat_generate_sparse_output(dir_Masks_output)
