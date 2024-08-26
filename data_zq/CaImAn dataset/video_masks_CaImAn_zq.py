# originated from paper_reproduction/utils/video_masks_CaImAn.m
# 0. find tiff files and order them
# 1. load movies in the mov h5 file
# 2. load the regions (training data only)
# 0 & 1 ignored, only generate GT_masks here

import json
import numpy as np
from scipy.io import savemat

dir_data_file = "D:\\0_Project\OBMI_Data\\20230323_CalmAn\WEBSITE"
# list_caiman = ['J115', 'J123', 'K53', 'YST']
list_caiman = ['YST']
# lateral dimensions to crop four sub-videos
xyrange = [[1, 224, 240, 463, 1, 224, 249, 472],
           [1, 152, 169, 320, 1, 216, 243, 458],
           [1, 248, 265, 512, 1, 248, 265, 512],
           [1, 88, 113, 200, 1, 120, 137, 256]]

for ind in range(len(list_caiman)):
    data_name = list_caiman[ind]

    with open(f'{dir_data_file}/{data_name}/regions/consensus_regions.json', 'r') as f:
        regions = json.load(f)
    num_masks = len(regions)

    with open(f'{dir_data_file}/{data_name}/info.json', 'r') as f:
        info = json.load(f)
    dimensions = info['dimensions']
    h, w = dimensions[1], dimensions[2]  # "dimensions": [3000, 200, 256], [frames, height, width]


    # # simplified
    # masks_tmp = []
    # for s in regions:
    #     mask = np.zeros((h, w))
    #     coords = s['coordinates']
    #     mask[tuple(zip(*coords))] = 1
    #     masks_tmp.append(mask)
    # masks = np.array(masks_tmp)

    # v2
    def tomask(coords):
        mask = np.zeros((h, w))
        mask[tuple(zip(*coords))] = 1
        return mask


    masks = np.array([tomask(s['coordinates']) for s in regions]).astype('uint8')

    areas = np.sum(np.sum(masks, axis=2), axis=1)  # 按照行，再按列相加

    ind = 3
    # w, h = 463, 472
    # attention the index in python starts from 0 while that in matlab starts from 1
    for xpart in range(1, 3):  # h
        for ypart in range(1, 3):  # w
            xrange_1 = xyrange[ind][2 * xpart - 1 - 1]
            xrange_2 = xyrange[ind][2 * xpart - 1]
            yrange_1 = xyrange[ind][2 * ypart - 1 + 4 - 1]
            yrang_2 = xyrange[ind][2 * ypart + 4 - 1]
            FinalMasks = masks[:, xrange_1 - 1:xrange_2, yrange_1 - 1:yrang_2]
            areas_cut = np.sum(np.sum(FinalMasks, axis=2), axis=1)
            areas_ratio = areas_cut / areas
            frames_to_keep = areas_ratio >= 1 / 3
            true_count = np.sum(frames_to_keep)
            print("Number of frames to keep:", true_count)
            FinalMasks = FinalMasks[frames_to_keep, :, :]
            FinalMasks = FinalMasks.transpose([0, 2, 1])
            # FinalMasks[areas_ratio < 1 / 3, :, :] = []
            mask_name = f"./{data_name}/GT Masks/FinalMasks_{data_name}_part{xpart}{ypart}.mat"
            savemat(mask_name, {'FinalMasks': FinalMasks}, format='5')
