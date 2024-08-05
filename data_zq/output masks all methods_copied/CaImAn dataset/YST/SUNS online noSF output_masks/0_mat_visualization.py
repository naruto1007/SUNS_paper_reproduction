import os
import h5py
import numpy as np
from scipy.io import loadmat


def mat_visualization_GT(filename):
    mat = h5py.File(filename, 'r')
    FinalMasks = np.array(mat["FinalMasks"]).transpose([1, 2, 0])
    mat.close()
    print("FinalMasks.shape: ", FinalMasks.shape)
    data_GT = FinalMasks[:, :, 3]
    coordinates = np.where(data_GT == 1)
    print("最大值1的坐标：", list(zip(coordinates[0], coordinates[1])))
    print(data_GT)


def mat_visualization_output(filename):
    try:
        mat = h5py.File(filename, 'r')
        Masks = np.array(mat['Masks'])
        mat.close()
    except OSError:
        mat = loadmat(filename)
        Masks = np.array(mat["Masks"]).transpose([1, 2, 0])
    print("Masks.shape: ", Masks.shape)
    data_output = Masks[:, :, 3]
    coordinates = np.where(data_output == 1)
    print("最大值1的坐标：", list(zip(coordinates[0], coordinates[1])))
    print(data_output)


dir_Masks_GT = "D:\PyCharm_project\SUNS_paper_reproduction\data_zq\CaImAn dataset\YST\GT Masks"
filename_GT = os.path.join(dir_Masks_GT, "FinalMasks_YST_part11.mat")
mat_visualization_GT(filename_GT)

# dir_Masks_output = "D:\PyCharm_project\SUNS_paper_reproduction\data_zq\output masks all methods_copied\CaImAn dataset\YST\SUNS noSF output_masks"
# filename_output = os.path.join(dir_Masks_output, "Output_Masks_YST_part11.mat")

filename_output = "Output_Masks_YST_part11.mat"

mat_visualization_output(filename_output)
