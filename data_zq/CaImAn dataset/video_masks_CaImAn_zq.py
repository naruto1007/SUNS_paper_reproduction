# originated from paper_reproduction/utils/video_masks_CaImAn.m
# 0. find tiff files and order them
# 1. load movies in the mov h5 file
# 2. load the regions (training data only)
# 0 & 1 ignored, only generate GT_masks here

import json
import numpy as np
from scipy.io import savemat

dir_data_file = "D:\\0_Project\OBMI_Data\\20230323_CalmAn\WEBSITE"
list_caiman = ['J115', 'J123', 'K53', 'YST']
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
    w, h = dimensions[2], dimensions[1]  # "dimensions": [3000, 200, 256], [frames, height, width]

    # mask = np.zeros((w, h), dtype=bool)
    # masks = np.zeros((w, h, num_masks), dtype=bool)
    # for i in range(num_masks):
    #     if isinstance(regions[i], dict):
    #         coords = np.array(regions[i]['coordinates']) + 2
    #     elif isinstance(regions[i], list):
    #         coords = np.array(regions[i]['coordinates']) + 2
    #     else:
    #         raise ValueError("Unsupported region type")
    #
    #     mask = np.zeros((w, h), dtype=bool)
    #     # mask[tuple(np.transpose(coords))] = True
    #     for x, y in coords:
    #         mask[y, x] = 1
    #     masks[:, :, i] = mask

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

    masks = np.array([tomask(s['coordinates']) for s in regions])

    areas = np.sum(np.sum(masks, axis=1), axis=0)

    # w, h = 463, 472
    for xpart in range(1, 3):
        for ypart in range(1, 3):
            xrange = [xyrange[ind][2 * xpart - 1 - 1], xyrange[ind][2 * xpart - 1]]
            yrange = [xyrange[ind][2 * ypart - 1 + 4 - 1], xyrange[ind][2 * ypart + 4 - 1]]
            FinalMasks = masks[xrange, yrange, :]
            areas_cut = np.sum(np.sum(FinalMasks, axis=1), axis=0)
            areas_ratio = areas_cut / areas
            FinalMasks[:, :, areas_ratio < 1 / 3] = False
            mask_name = f"./{data_name}/GT Masks/FinalMasks_{data_name}_part{xpart}{ypart}.mat"
            savemat(mask_name, {'FinalMasks': FinalMasks}, format='5')
