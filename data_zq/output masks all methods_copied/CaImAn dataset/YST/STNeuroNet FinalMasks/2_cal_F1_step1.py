# calculate all metrics for every kind of data, saved in Output_Info_All_cal.mat
# the mat file includes list_Recall, list_Precision, list_F1, list_time,list_time_frame
# a list_F1 includes n scores, n=6 or n=10 maybe
import numpy as np
import os
from scipy.io import savemat, loadmat
from suns.PostProcessing.evaluate import GetPerformance_Jaccard_2


def Performance_Cal_Suns_noSF(root_dir, result_dir, data_type_1, data_type_2, list_Exp_ID, method_type):
    # method: SUNS noSF output_masks
    nvideo = len(list_Exp_ID)
    list_CV = list(range(0, nvideo))
    num_CV = len(list_CV)
    list_Recall = np.zeros((num_CV, 1))
    list_Precision = np.zeros((num_CV, 1))
    list_F1 = np.zeros((num_CV, 1))
    list_time = np.zeros((num_CV, 4))
    list_time_frame = np.zeros((num_CV, 4))

    # dir_Masks_GT = "D:\PyCharm_project\SUNS_paper_reproduction\data_zq\CalmAn_dataset\YST\GT Masks"
    # dir_Masks_output = "D:\PyCharm_project\SUNS_paper_reproduction\data_zq\output masks all methods_copied\CaImAn dataset\YST\SUNS noSF output_masks"
    # dir_GTMasks = os.path.join(dir_Masks_GT, 'FinalMasks_')
    # dir_OutputMasks = os.path.join(dir_Masks_output, 'Output_Masks_')

    dir_GTMasks = os.path.join(root_dir, data_type_1, data_type_2, "GT Masks", 'FinalMasks_')
    dir_Masks_output = os.path.join(root_dir, result_dir, data_type_1, data_type_2, method_type)
    dir_OutputMasks = os.path.join(dir_Masks_output, 'FinalSegments_')

    time_total = 0
    time_frame = 0

    for CV in list_CV:
        Exp_ID = list_Exp_ID[CV]
        print('Video ', Exp_ID)
        # GT_sparse
        filename_GT = dir_GTMasks + Exp_ID + '_sparse_generated.mat'
        data_GT = loadmat(filename_GT)
        GTMasks_2 = data_GT['GTMasks_2'].transpose()
        # Mask_sparse
        filename_output = dir_OutputMasks + Exp_ID + '_sparse.mat'
        data_Masks = loadmat(filename_output)
        Masks_2 = data_Masks['Masks'].transpose()
        # calculate
        (Recall, Precision, F1) = GetPerformance_Jaccard_2(GTMasks_2, Masks_2, ThreshJ=0.5)
        print({'Recall': Recall, 'Precision': Precision, 'F1': F1})

        list_Recall[CV] = Recall
        list_Precision[CV] = Precision
        list_F1[CV] = F1
        list_time[CV] = time_total
        list_time_frame[CV] = time_frame

        Info_dict = {'list_Recall': list_Recall, 'list_Precision': list_Precision, 'list_F1': list_F1,
                     'list_time': list_time, 'list_time_frame': list_time_frame}
        savemat(os.path.join(dir_Masks_output, 'Output_Info_All_cal.mat'), Info_dict)


root_dir = "D:\PyCharm_project\SUNS_paper_reproduction\data_zq"
result_dir = "output masks all methods_copied"

data_type_1 = "CaImAn dataset"
data_type_2 = "YST"
method_type = "STNeuroNet FinalMasks"

list_Exp_ID = ['YST_part11', 'YST_part12', 'YST_part21', 'YST_part22']

Performance_Cal_Suns_noSF(root_dir, result_dir, data_type_1, data_type_2, list_Exp_ID, method_type)
