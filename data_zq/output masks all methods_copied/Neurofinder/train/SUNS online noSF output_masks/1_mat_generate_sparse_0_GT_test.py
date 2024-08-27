# test the method used for generate sparse right or not
# if three steps is all yes, the test above is right
import os
import h5py
import numpy as np
from scipy.io import loadmat, savemat

dir_Masks_GT = "D:\PyCharm_project\SUNS_paper_reproduction\data_zq\\Neurofinder\\train\GT Masks"
filename_output = os.path.join(dir_Masks_GT, "FinalMasks_04.01_sparse_generated.mat")
data_Masks = loadmat(filename_output)
Masks_2 = data_Masks['GTMasks_2'].transpose()

# generate Masks to store as done in suns_batch
Lx = 480
Ly = 416
Masks = np.reshape(Masks_2.toarray(), (Masks_2.shape[0], Lx, Ly)).astype('bool')
savemat(os.path.join(dir_Masks_GT, "FinalMasks_04.01_generated_from_sparse.mat"), {'Masks': Masks},
        do_compression=True)

# compare the two mat is the same
filename_GT = os.path.join(dir_Masks_GT, "FinalMasks_04.01.mat")
mat = h5py.File(filename_GT, 'r')
FinalMasks = np.array(mat["FinalMasks"]).transpose([2, 1, 0]).astype('int32')
mat.close()

output_generated_from_sparse = os.path.join(dir_Masks_GT, "FinalMasks_04.01_generated_from_sparse.mat")
data_output_generated_from_sparse = loadmat(output_generated_from_sparse)
masks_output_generated_from_sparse = data_output_generated_from_sparse['Masks'].transpose([1, 2, 0])

# nums of 1 is equal?
num_1_original = np.count_nonzero(FinalMasks == 1)
num_1_output_generated_from_sparse = np.count_nonzero(masks_output_generated_from_sparse == 1)
print("num_original:", num_1_original)
print("num_output_generated_from_sparse:", num_1_output_generated_from_sparse)

result = np.array_equal(FinalMasks, masks_output_generated_from_sparse)
# result = np.array_equal(masks_output_generated_from_sparse, Masks)
print(result)
