# test whether output sparse masks transfer succeed or not
import os
import numpy as np
from scipy.io import savemat, loadmat

# generate Masks_2 used in GetPerformance_Jaccard_2
dir_Masks_output = "D:\PyCharm_project\SUNS_paper_reproduction\data_zq\output masks all methods_copied\CaImAn dataset\YST\SUNS noSF output_masks"
filename_output = os.path.join(dir_Masks_output, "Output_Masks_YST_part11_sparse.mat")
data_Masks = loadmat(filename_output)
Masks_2 = data_Masks['Masks'].transpose()

# generate Masks to store as done in suns_batch
Lx = 120
Ly = 88
Masks = np.reshape(Masks_2.toarray(), (Masks_2.shape[0], Lx, Ly)).astype('bool')
savemat(os.path.join(dir_Masks_output, "Output_Masks_YST_part11_generated_from_sparse.mat"), {'Masks': Masks},
        do_compression=True)

# compare the two mat is the same
output_original = os.path.join(dir_Masks_output, "Output_Masks_YST_part11.mat")
data_output_original = loadmat(output_original)
masks_output_original = data_output_original['Masks']

output_generated_from_sparse = os.path.join(dir_Masks_output, "Output_Masks_YST_part11_generated_from_sparse.mat")
data_output_generated_from_sparse = loadmat(output_generated_from_sparse)
masks_output_generated_from_sparse = data_output_generated_from_sparse['Masks']

# nums of 1 is equal?

result = np.array_equal(masks_output_original, masks_output_generated_from_sparse)
# result = np.array_equal(masks_output_generated_from_sparse, Masks)
print(result)
