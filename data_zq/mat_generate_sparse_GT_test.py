# test the method used for generate sparse right or not
# if three steps is all yes, the test above is right
import os
import numpy as np
from scipy.io import loadmat

dir_Masks_GT = "D:\PyCharm_project\SUNS_paper_reproduction\data_zq\CalmAn_dataset\YST\GT Masks"
# 0.generate GT_sparse
# have done

# 1.test if the GT_sparse_generated is the same as the GT_sparse_given
# _original is the same, _generated is not the same, need to check the sparse_generated function in mat_generate_sparse.py
filename_GT_sparse_generated = os.path.join(dir_Masks_GT, "FinalMasks_YST_part11_sparse_original.mat")
# filename_GT_sparse_generated = os.path.join(dir_Masks_GT, "FinalMasks_YST_part11_sparse_generated.mat")
filename_GT_sparse_given = os.path.join(dir_Masks_GT, "FinalMasks_YST_part11_sparse.mat")

data_GT_sparse_generated = loadmat(filename_GT_sparse_generated)
data_GT_sparse_given = loadmat(filename_GT_sparse_given)

Masks_GT_sparse_generated = data_GT_sparse_generated['GTMasks_2']
Masks_GT_sparse_given = data_GT_sparse_given['GTMasks_2']

result = np.array_equal(Masks_GT_sparse_generated.toarray(), Masks_GT_sparse_given.toarray())
print("The GT_sparse_generated is the same as the GT_sparse_given: ", result)

# 2.test if mat_from_GT_sparse_given is the same as the GT.mat



# 3.test if mat_from_GT_sparse_generated is the same as the GT.mat

