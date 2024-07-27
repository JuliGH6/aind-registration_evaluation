from main_qa import (EvalStitching,get_default_config)
import time
from aind_registration_evaluation.util.file_conversion import n5_to_tiff, n5_to_tiff_warped, apply_mask, apply_mask_and_crop, downsample_and_int16, prepareFiles, normalize_image
from aind_registration_evaluation.util.excelOutput import create_excel_file, append_results_to_excel
import os
import numpy as np
import pandas as pd
import multiprocessing
from dask.distributed import Client, LocalCluster
import itertools 
from scipy.ndimage import affine_transform
import tifffile as tiff



def calculate_roi(image1,image2,matrix):

    default_config = {
        "metrics": ["ncc", "mi", "nmi"],
        "transform_matrix": matrix,
    }
    print("\n\n\n\n\n", image1, image2)
    default_config["image_1"] = image1
    default_config["image_2"] = image2

    mod = EvalStitching(default_config)

    results = mod.run_roi(0.008, 0.05, 5, 0.3)
    matching_value = (results['num_matching_cells'] ** 2) / (results['num_img1_cells'] * results['num_img2_cells'])
    return (results['nmi']['weighted_avg'], matching_value)



def calculate_full_image(image1,image2):

    default_config = {
        "image_channel": 0,
        "data_type": "small",
        "metrics": ["ncc", "nmi"],
        #"ksam" is not implemented
        "window_size": 30,
        "sampling_info": {
            "sampling_type": "grid",
            "numpoints": 60
        },
        "visualize": False,
        "transform_matrix": [
            [1, 0, 0, 0],  # Z
            [0, 1, 0, 0],  # Y
            [0, 0, 1, 0],  # X
            [0, 0, 0, 1]
        ]
    }
    default_config["image_1"] = image1
    default_config["image_2"] = image2


    mod = EvalStitching(default_config)
    results = mod.run()

    ncc = np.mean(results['ncc']["point_metric"])
    nmi = np.mean(results['nmi']["point_metric"])

    return (ncc,nmi)

def get_valid_combinations():
    values = np.linspace(0.1, 0.6, 5)  

    combinations = []
    for alpha, beta, gamma, delta in itertools.product(values, repeat=4):
        combinations.append((alpha, beta, gamma, delta))
    print(combinations)
    return combinations

def find_min_max_across_arrays(*arrays):
    transposed_arrays = np.array(arrays).T  # Transpose arrays to work with indices
    min_values = []
    max_values = []
    
    for index_values in transposed_arrays:
        min_values.append(np.min(index_values))
        max_values.append(np.max(index_values))
    
    return min_values, max_values

def normalize(*arrays):
    transposed_arrays = np.array(arrays).T  # Transpose arrays to work with indices
    min_values, max_values = find_min_max_across_arrays(*arrays)
    
    normalized_transposed_arrays = []
    
    for i, index_values in enumerate(transposed_arrays):
        min_val = min_values[i]
        max_val = max_values[i]
        
        if max_val != min_val:
            normalized_index_values = [(val - min_val) / (max_val - min_val) for val in index_values]
        else:
            normalized_index_values = list(index_values)
        
        normalized_transposed_arrays.append(normalized_index_values)
    
    normalized_arrays = np.array(normalized_transposed_arrays).T.tolist()  # Transpose back to the original shape
    
    return [list(arr) for arr in normalized_arrays]

def main():
    n_workers = multiprocessing.cpu_count()
    threads_per_worker = 1
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=True,
        memory_limit="auto",
    )

    client = Client(cluster)

    matrizes = [
            [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ],
        [
            [1, 0, 0, 8],
            [0, 1, 0, 8],
            [0, 0, 1, 8],
            [0, 0, 0, 1]
        ],
        [
            [1, 0, 0, 16],
            [0, 1, 0, 16],
            [0, 0, 1, 16],
            [0, 0, 0, 1]
        ],
        [
            [1, 0, 0, 32],
            [0, 1, 0, 32],
            [0, 0, 1, 32],
            [0, 0, 0, 1]
        ],
        [
            [1, 0, 0, 64],
            [0, 1, 0, 64],
            [0, 0, 1, 64],
            [0, 0, 0, 1]
        ],
    ]

    combinations = get_valid_combinations()
    imageName = '/root/capsule/scratch/confocal_s0_cropped_dsLev1.tif'#change trans and in mainqa
    image = tiff.imread(imageName)
    expectedResult = []
    matching_mask_ROI_nmi_list = []
    matching_value_list= []
    full_image_ncc_list= []
    full_image_nmi_list= []
    for i,matrix in enumerate(matrizes):
        trans_image = affine_transform(image, matrix)
        trans_image_name = '/root/capsule/scratch/confocal_s0_cropped_dsLev1_trans'+str(i)+'.tif'
        tiff.imwrite(trans_image_name, trans_image)
        
        expectedResult.append(1 - i*0.24)
        
        matching_mask_ROI_nmi, matching_value = calculate_roi(imageName, trans_image_name, matrix)
        full_image_ncc, full_image_nmi = calculate_full_image(imageName, trans_image_name)

        if np.isnan(matching_mask_ROI_nmi) or np.isnan(matching_value) or np.isnan(full_image_ncc) or np.isnan(full_image_nmi): continue

        matching_mask_ROI_nmi_list.append(matching_mask_ROI_nmi)
        matching_value_list.append(matching_value)
        full_image_ncc_list.append(full_image_ncc)
        full_image_nmi_list.append(full_image_nmi)

    print(matching_mask_ROI_nmi_list)
    print(matching_value_list)
    print(full_image_ncc_list)
    print(full_image_nmi_list)

    norm_matching_mask_ROI_nmi_list,norm_matching_value_list,norm_full_image_ncc_list,norm_full_image_nmi_list = normalize(matching_mask_ROI_nmi_list,matching_value_list , full_image_ncc_list,full_image_nmi_list)

    print(norm_matching_mask_ROI_nmi_list)
    print(norm_matching_value_list)
    print(norm_full_image_ncc_list)
    print(norm_full_image_nmi_list)

    bestCombs = []
    leastErrors = []
    for i in range(len(norm_full_image_ncc_list)):
        bestCombination = None
        leastDifference = float('inf')
        for comb in combinations:
            calculated_value = comb[0] * norm_matching_value_list[i] + comb[1]*norm_full_image_ncc_list[i] + comb[2]*norm_full_image_nmi_list[i] + comb[3]*norm_matching_mask_ROI_nmi_list[i]
            difference = abs(expectedResult[i] - calculated_value)

            if difference < leastDifference:
                bestCombination = comb
                leastDifference = difference
        bestCombs.append(bestCombination)
        leastErrors.append(leastDifference)
        print("BestComb:", bestCombination, "LeastError:", leastDifference)

    a_s = 0
    b_s = 0
    c_s = 0
    d_s = 0
    for i, comb in enumerate(bestCombs):
        a,b,c,d = comb
        a_s += a *leastErrors[i]
        b_s += b *leastErrors[i]
        
        c_s += c *leastErrors[i]
        d_s += d *leastErrors[i]

    totalError = sum(leastErrors)
    a_s = a_s/totalError
    b_s = b_s/totalError
    c_s = c_s/totalError
    d_s = d_s/totalError

    l = len(bestCombs)
    finalWeights = (a_s/l, b_s/l, c_s/l, d_s/l)

    print(finalWeights)

    client.close()

if __name__ == "__main__":
    main()