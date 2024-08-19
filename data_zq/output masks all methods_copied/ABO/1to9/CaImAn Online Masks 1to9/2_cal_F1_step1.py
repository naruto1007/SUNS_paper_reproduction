# calculate all metrics for every kind of data, saved in Output_Info_All_cal.mat
# the mat file includes list_Recall, list_Precision, list_F1, list_time,list_time_frame
# a list_F1 includes n scores, n=6 or n=10 maybe
import numpy as np
import os
from scipy.io import savemat, loadmat
from suns.PostProcessing.evaluate import GetPerformance_Jaccard_2


def Performance_Cal(root_dir, result_dir, data_type_1, data_type_2, data_type_3, list_Exp_ID, list_Exp_ID_test,
                    method_type):
    # method: SUNS noSF output_masks
    nvideo = len(list_Exp_ID)
    list_CV = list(range(0, nvideo))
    num_CV = len(list_CV)
    list_Recall = np.zeros((num_CV, 1))
    list_Precision = np.zeros((num_CV, 1))
    list_F1 = np.zeros((num_CV, 1))
    list_time = np.zeros((num_CV, 4))
    list_time_frame = np.zeros((num_CV, 4))

    dir_GTMasks = os.path.join(root_dir, data_type_1, data_type_2, "GT Masks", 'FinalMasks_FPremoved_')
    dir_Masks_output = os.path.join(root_dir, result_dir, data_type_1, data_type_3, method_type)

    time_total = 0
    time_frame = 0

    for CV in list_CV:
        Exp_ID = list_Exp_ID[CV]
        print('Video ', Exp_ID)
        dir_OutputMasks = os.path.join(dir_Masks_output, Exp_ID, '')

        Recall_test = np.zeros((num_CV, 1))
        Precision_test = np.zeros((num_CV, 1))
        F1_test = np.zeros((num_CV, 1))
        for CV_test in list_CV:
            if CV_test != CV:
                Exp_ID_test = list_Exp_ID_test[CV_test]
                # GT_sparse
                filename_GT = dir_GTMasks + Exp_ID_test + '_sparse_generated.mat'
                data_GT = loadmat(filename_GT)
                GTMasks_2 = data_GT['GTMasks_2'].transpose()
                # Mask_sparse
                filename_output = dir_OutputMasks + Exp_ID_test + '_neurons_sparse.mat'
                data_Masks = loadmat(filename_output)
                Masks_2 = data_Masks['Masks'].transpose()
                # calculate
                (Recall_test[CV_test], Precision_test[CV_test], F1_test[CV_test]) = GetPerformance_Jaccard_2(GTMasks_2,
                                                                                                             Masks_2,
                                                                                                             ThreshJ=0.5)

        # cal mean value for 1 to 9
        Recall = sum(Recall_test) / (num_CV - 1)
        Precision = sum(Precision_test) / (num_CV - 1)
        F1 = sum(F1_test) / (num_CV - 1)
        time_total = 0
        time_frame = 0
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

data_type_1 = "ABO"
data_type_2 = "275"
data_type_3 = "1to9"
method_type = "CaImAn Online Masks 1to9"

list_Exp_ID = ['CV0', 'CV1', 'CV2', 'CV3', 'CV4', 'CV5', 'CV6', 'CV7', 'CV8', 'CV9']

# old version
# list_Exp_ID_test = ['501484643', '501574836', '501729039', '502608215', '503109347', '510214538', '524691284',
#                     '527048992', '531006860', '539670003']

# new order here
list_Exp_ID_test = ['524691284', '531006860', '502608215', '503109347', '501484643', '501574836', '501729039',
                    '539670003', '510214538', '527048992']

Performance_Cal(root_dir, result_dir, data_type_1, data_type_2, data_type_3, list_Exp_ID, list_Exp_ID_test, method_type)
