# test whether output sparse masks transfer succeed or not
import os
import numpy as np
import h5py
from scipy.io import savemat, loadmat

# generate Masks_2 used in GetPerformance_Jaccard_2
dir_Masks_output = "D:\PyCharm_project\SUNS_paper_reproduction\data_zq\output masks all methods_copied\\Neurofinder\\train\SUNS noSF output_masks"
filename_output = os.path.join(dir_Masks_output, "Output_Masks_04.01_sparse.mat")
data_Masks = loadmat(filename_output)
Masks_2 = data_Masks['Masks'].transpose()

# generate Masks to store as done in suns_batch
Lx = 480
Ly = 416
Masks = np.reshape(Masks_2.toarray(), (Masks_2.shape[0], Lx, Ly)).astype('bool')
savemat(os.path.join(dir_Masks_output, "Output_Masks_04.01_generated_from_sparse.mat"), {'Masks': Masks},
        do_compression=True)

# compare the two mat is the same
output_original = "Output_Masks_04.01.mat"
mat = h5py.File(output_original, 'r')
masks_output_original = np.array(mat["Masks"])

output_generated_from_sparse = os.path.join(dir_Masks_output, "Output_Masks_04.01_generated_from_sparse.mat")
data_output_generated_from_sparse = loadmat(output_generated_from_sparse)
masks_output_generated_from_sparse = data_output_generated_from_sparse['Masks'].transpose([1, 2, 0])

# nums of 1 is equal?
num_1_original = np.count_nonzero(masks_output_original == 1)
num_1_output_generated_from_sparse = np.count_nonzero(masks_output_generated_from_sparse == 1)
print("num_original:", num_1_original)
print("num_output_generated_from_sparse:", num_1_output_generated_from_sparse)

result = np.array_equal(masks_output_original, masks_output_generated_from_sparse)
# result = np.array_equal(masks_output_generated_from_sparse, Masks)
print(result)
