from main_qa import (EvalStitching,get_default_config)
from aind_registration_evaluation.util.file_conversion import n5_to_tiff, n5_to_tiff_warped, apply_mask, apply_mask_and_crop, downsample_and_int16, prepareFiles, normalize_image
from aind_registration_evaluation.util.excelOutput import create_excel_file, append_results_to_excel
import os
import numpy as np
import pandas as pd
import multiprocessing
from dask.distributed import Client, LocalCluster
import time


#check ReadMe for options
config_dict = {
    "image_channel": 0,
    "data_type": "small",
    "metrics": ["ncc", "mi", "nmi"],
    #"ksam" is not implemented
    "window_size": 20,
    "sampling_info": {
        "sampling_type": "grid",
        "numpoints": 80
    },
    "visualize": False,
}

def main():

    # Get same configuration from yaml file to apply it over a dataset
    default_config = config_dict

    # confocal = "/data/coregistration_test_693124_06-21-2024/confocal/693124_confocal_0_300_20xWater_stitched.n5"
    # warpedCortical = "/data/coreg_test_693124_cortical_z_stack_to_0-300.n5"
    # prepareFiles(confocal, warpedCortical, 1, 1)

    n_workers = multiprocessing.cpu_count()
    threads_per_worker = 1
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=True,
        memory_limit="auto",
    )

    client = Client(cluster)


    default_config["image_1"] = '/root/capsule/scratch/conf09.tif'
    default_config["image_2"] = '/root/capsule/scratch/cort09.tif'




    default_config["transform_matrix"] = [
        [1, 0, 0, 0],  # Z
        [0, 1, 0, 0],  # Y
        [0, 0, 1, 0],  # X
        [0, 0, 0, 1],
    ]

    mod = EvalStitching(default_config)
    time_start = time.time()
    mod.run()
    time_end = time.time()
    duration = time_end-time_start
    print(f"Time: {duration}")

    # window_sizes = [30]
    # num_of_points = [60]

    # # corticals = ['/scratch/cortical_s0_cropped_downsampled_f1.tif', '/scratch/cortical_s0_cropped_downsampled_f2.tif']
    # # confocals = ['/scratch/confocal_s0_cropped_downsampled_f1.tif', '/scratch/confocal_s0_cropped_downsampled_f2.tif']
    # confocals = ['/root/capsule/data/693124_coregistration/confocal_s0_cropped_dsLev1.tif']
    # corticals = ['/root/capsule/data/693124_coregistration/cortical_s0_cropped_dsLev1.tif']


    # matrizes = [
    #     [
    #         [1, 0, 0, 0],  # Z
    #         [0, 1, 0, 0],  # Y
    #         [0, 0, 1, 0],  # X
    #         [0, 0, 0, 1],
    #     ]
    #     # [
    #     #     [1, 0, 0, 4],
    #     #     [0, 1, 0, 4],
    #     #     [0, 0, 1, 4],
    #     #     [0, 0, 0, 1]
    #     # ],
    #     # [
    #     #     [1, 0, 0, 8],
    #     #     [0, 1, 0, 8],
    #     #     [0, 0, 1, 8],
    #     #     [0, 0, 0, 1]
    #     # ],
    #     # [
    #     #     [1, 0, 0, 12],
    #     #     [0, 1, 0, 12],
    #     #     [0, 0, 1, 12],
    #     #     [0, 0, 0, 1]
    #     # ],
    #     # [
    #     #     [1, 0, 0, 16],
    #     #     [0, 1, 0, 16],
    #     #     [0, 0, 1, 16],
    #     #     [0, 0, 0, 1]
    #     # ]
    #     # [
    #     #     [1, 0, 0, 20],  # Z
    #     #     [0, 1, 0, 20],  # Y
    #     #     [0, 0, 1, 20],  # X
    #     #     [0, 0, 0, 1],
    #     # ],
    #     # [
    #     #     [1, 0, 0, 24],
    #     #     [0, 1, 0, 24],
    #     #     [0, 0, 1, 24],
    #     #     [0, 0, 0, 1]
    #     # ],
    #     # [
    #     #     [1, 0, 0, 28],
    #     #     [0, 1, 0, 28],
    #     #     [0, 0, 1, 28],
    #     #     [0, 0, 0, 1]
    #     # ],
    # ]

    # # for matrix in matrizes:
    # #     default_config["transform_matrix"] = matrix
    # #     joined_rows = [",".join([str(value) for value in row]) for row in matrix]
    # #     excelMatrix = "|".join(joined_rows) 
    # #     for i in range(len(corticals)):
    # #         default_config["image_1"] = confocals[i]
    # #         default_config["image_2"] = corticals[i]
    # #         for ws in window_sizes:
    # #             for nump in num_of_points:
    # #                 default_config['window_size'] = ws
    # #                 default_config['sampling_info']['numpoints'] = nump

    # #                 mod = EvalStitching(default_config)

    # #                 time_start = time.time()
    # #                 results = mod.run()
    # #                 time_end = time.time()
    # #                 duration = time_end-time_start
    # #                 print(f"Time: {duration}")

    # #                 excel_results = [{
    # #                     "WindowSize": ws,
    # #                     "NumPoints": nump,
    # #                     "Runtime": duration,
    # #                     "Datatype": 'int16',
    # #                     "PointSampling": default_config['sampling_info']['sampling_type'],
    # #                     "Downsampling Factor": '2**1', #+ str(i+1),
    # #                     "Matrix": excelMatrix
    # #                 }]

    # #                 for n in default_config['metrics']:
    # #                     excel_results[0][n] = np.mean(results[n]["point_metric"])

    # #                 file_path = '/scratch/results_t.xlsx'
    # #                 create_excel_file(file_path)
    # #                 append_results_to_excel(file_path, excel_results)

    # if not os.path.exists("/scratch/results_roi_m.xlsx"):
    #     labels = ["NumberROIs", "MaxCentroidDistance", "Runtime", "Downsampling Factor", "Matrix", "ncc", "mi", "nmi"]
    #     initial_data = pd.DataFrame(columns=labels)
    #     initial_data.to_excel("/scratch/results_roi_m.xlsx", index=False)

    # maxCentroidDistance = [5]

    # roi_dict = {"metrics": ["ncc", "mi", "nmi"]}

    # for matrix in matrizes:
    #     roi_dict["transform_matrix"] = matrix
    #     joined_rows = [",".join([str(value) for value in row]) for row in matrix]
    #     excelMatrix = "|".join(joined_rows)        
    #     for i in range(len(corticals)):
    #         roi_dict["image_1"] = confocals[i]
    #         roi_dict["image_2"] = corticals[i]
    #         for mCd in maxCentroidDistance:

    #                 mod = EvalStitching(roi_dict)

    #                 time_start = time.time()
    #                 # results = mod.run_roi(0.008, 0.05, mCd, 0.3)
    #                 results = mod.run_random_patches(mCd)
    #                 time_end = time.time()
    #                 duration = time_end-time_start
    #                 print(f"Time: {duration}")

    #                 excel_results = [{
    #                     "NumberROIs": results["num_rois"],
    #                     "MaxCentroidDistance": mCd,
    #                     "Runtime": duration,
    #                     "Downsampling Factor": '2**1', #+ str(i+1),
    #                     "Matrix": excelMatrix
    #                 }]

    #                 for n in roi_dict['metrics']:
    #                     excel_results[0][n] = np.mean(results[n]["point_metric"])
            

    #                 file_path = '/scratch/results_roi_m.xlsx'
    #                 append_results_to_excel(file_path, excel_results)   

    client.close()



if __name__ == "__main__":
    main()

