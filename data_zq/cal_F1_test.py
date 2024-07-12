# test F1 score cal with provided GT and masks gegerated by suns
# explained in cal_F1.png
import os
import glob
import h5py
import numpy as np
from scipy.io import savemat, loadmat
from scipy import sparse

from suns.PostProcessing.evaluate import GetPerformance_Jaccard_2


def generate_sparse_mat(dir_Masks):
    dir_all = glob.glob(os.path.join(dir_Masks, '*Output_Masks*.mat'))
    for path_name in dir_all:
        file_name = os.path.split(path_name)[1]
        if '_sparse' not in file_name:
            print(file_name)
            try:  # If file_name is saved in '-v7.3' format
                mat = h5py.File(path_name, 'r')
                Masks = np.array(mat['Masks']).astype('bool')
                mat.close()
            except OSError:  # If file_name is not saved in '-v7.3' format
                mat = loadmat(path_name)
                Masks = np.array(mat["Masks"]).transpose([2, 1, 0]).astype('bool')

            (ncells, Ly, Lx) = Masks.shape
            Masks_2 = sparse.coo_matrix(Masks.reshape(ncells, Lx * Ly).T)
            savemat(os.path.join(path_name[:-4] + '_sparse.mat'), \
                    {'Masks': Masks_2}, do_compression=True)

# # generate sparse mat for output masks
# dir_Masks = "D:\PyCharm_project\SUNS_paper_reproduction\paper_reproduction\output masks all methods\CaImAn dataset\YST\SUNS noSF output_masks"
# generate_sparse_mat(dir_Masks)


# GT_sparse
filename_GT = "D:\PyCharm_project\SUNS_paper_reproduction\demo\data\GT Masks\FinalMasks_YST_part11_sparse.mat"
data_GT = loadmat(filename_GT)
GTMasks_2 = data_GT['GTMasks_2'].transpose()

# Mask_sparse
filename_Masks_sparse = "D:\PyCharm_project\SUNS_paper_reproduction\paper_reproduction\output masks all methods\CaImAn dataset\YST\SUNS noSF output_masks\Output_Masks_YST_part11_sparse.mat"
data_Masks = loadmat(filename_Masks_sparse)
Masks_2 = data_Masks['Masks'].transpose()

(Recall, Precision, F1) = GetPerformance_Jaccard_2(GTMasks_2, Masks_2, ThreshJ=0.5)
print({'Recall': Recall, 'Precision': Precision, 'F1': F1})
