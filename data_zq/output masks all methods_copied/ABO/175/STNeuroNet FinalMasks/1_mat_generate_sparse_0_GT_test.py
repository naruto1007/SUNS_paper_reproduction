# test the method used for generate sparse right or not
# if three steps is all yes, the test above is right
import os
import h5py
import numpy as np
from scipy.io import loadmat, savemat


def visualization(Masks):
    print(f"{Masks}.shape: {Masks.shape}")
    data_GT = Masks[:, :, 0]
    coordinates = np.where(data_GT == 1)
    print("最大值1的坐标：", list(zip(coordinates[0], coordinates[1])))
    print(data_GT)


dir_Masks_GT = "D:\PyCharm_project\SUNS_paper_reproduction\data_zq\ABO\\175\GT Masks"
filename_output = os.path.join(dir_Masks_GT, "FinalMasks_FPremoved_501271265_sparse_generated.mat")
data_Masks = loadmat(filename_output)
Masks_2 = data_Masks['GTMasks_2'].transpose()

# generate Masks to store as done in suns_batch
Lx = 487
Ly = 487
Masks = np.reshape(Masks_2.toarray(), (Masks_2.shape[0], Lx, Ly)).astype('bool')
savemat(os.path.join(dir_Masks_GT, "FinalMasks_FPremoved_501271265_generated_from_sparse.mat"), {'Masks': Masks},
        do_compression=True)

# compare the two mat is the same
output_original = os.path.join(dir_Masks_GT, "FinalMasks_FPremoved_501271265.mat")
mat = h5py.File(output_original, 'r')
masks_output_original = np.array(mat["FinalMasks"]).astype('uint8')
mat.close()
# visualization(masks_output_original)

output_generated_from_sparse = os.path.join(dir_Masks_GT, "FinalMasks_FPremoved_501271265_generated_from_sparse.mat")
data_output_generated_from_sparse = loadmat(output_generated_from_sparse)
masks_output_generated_from_sparse = data_output_generated_from_sparse['Masks']
# visualization(masks_output_generated_from_sparse)

# nums of 1 is equal?
num_1_original = np.count_nonzero(masks_output_original == 1)
num_1_output_generated_from_sparse = np.count_nonzero(masks_output_generated_from_sparse == 1)
print("num_original:", num_1_original)
print("num_output_generated_from_sparse:", num_1_output_generated_from_sparse)

result = np.array_equal(masks_output_original, masks_output_generated_from_sparse)
# result = np.array_equal(masks_output_generated_from_sparse, Masks)
print(result)
