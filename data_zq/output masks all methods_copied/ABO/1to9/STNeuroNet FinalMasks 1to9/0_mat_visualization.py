import os
import h5py
import numpy as np
from scipy.io import loadmat


def mat_visualization_GT(filename):
    try:
        mat = h5py.File(filename, 'r')
        FinalMasks = np.array(mat["FinalMasks"]).transpose([1, 2, 0]).astype('int')
        mat.close()
    except OSError:
        mat = loadmat(filename)
        FinalMasks = np.array(mat["FinalMasks"]).transpose([1, 2, 0]).astype('int')
    print("FinalMasks.shape: ", FinalMasks.shape)
    data_GT = FinalMasks[:, :, 3]
    coordinates = np.where(data_GT == 1)
    print("最大值1的坐标：", list(zip(coordinates[0], coordinates[1])))
    print(data_GT)


def mat_visualization_output(filename):
    try:
        mat = h5py.File(filename, 'r')
        Masks = np.array(mat['finalSegments'])
        mat.close()
    except OSError:
        mat = loadmat(filename)
        Masks = np.array(mat["finalSegments"])
    print("Masks.shape: ", Masks.shape)
    data_output = Masks[:, :, 3]
    coordinates = np.where(data_output == 1)
    print("最大值1的坐标：", list(zip(coordinates[0], coordinates[1])))
    print(data_output)


dir_Masks_GT = "D:\PyCharm_project\SUNS_paper_reproduction\data_zq\ABO\\275\GT Masks"
filename_GT = os.path.join(dir_Masks_GT, "FinalMasks_FPremoved_501574836.mat")
mat_visualization_GT(filename_GT)

# dir_Masks_output = "D:\PyCharm_project\SUNS_paper_reproduction\data_zq\output masks all methods_copied\ABO\\1to9\Suite2p Masks 1to9\CV0"
filename_output = os.path.join("./CV0", "FinalSegments_501574836.mat")

mat_visualization_output(filename_output)
