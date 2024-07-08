from main_qa import (EvalStitching,get_default_config)
import time
from aind_registration_evaluation.util.file_conversion import n5_to_tiff, n5_to_tiff_warped, apply_mask, apply_mask_and_crop, downsample_and_int16, prepareFiles, normalize_image
from aind_registration_evaluation.util.excelOutput import create_excel_summary, create_excel_file, append_results_to_excel
import os
import numpy as np
import pandas as pd
import multiprocessing
from dask.distributed import Client, LocalCluster


#check ReadMe for options
config_dict = {
    "image_channel": 0,
    "data_type": "small",
    "metrics": ["ssim", "ncc", "mi", "nmi", "issm"],
    #"ksam" is not implemented
    "window_size": 5,
    "sampling_info": {
        "sampling_type": "grid",
        "numpoints": 40
    },
    "visualize": True,
}

def main():

    # Get same configuration from yaml file to apply it over a dataset
    default_config = config_dict

    confocal = "/data/coregistration_test_693124_06-21-2024/confocal/693124_confocal_0_300_20xWater_stitched.n5"
    warpedCortical = "/data/coreg_test_693124_cortical_z_stack_to_0-300.n5"
    prepareFiles(confocal, warpedCortical, 1, 1)

    n_workers = multiprocessing.cpu_count()
    threads_per_worker = 1
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=True,
        memory_limit="auto",
    )

    client = Client(cluster)


    default_config["image_1"] = '/scratch/roi_cortical.tif'
    default_config["image_2"] = '/scratch/roi_confocal.tif'




    default_config["transform_matrix"] = [
        [1, 0, 0],  # Z
        [0, 1, 0],  # Y
        [0, 0, 1],  # X
    ]

    mod = EvalStitching(default_config)
    time_start = time.time()
    mod.run()
    time_end = time.time()
    duration = time_end-time_start
    print(f"Time: {duration}")

    # window_sizes = [30,60,100]
    # num_of_points = [100,200,400]
    # window_sizes = [15,50]
    # num_of_points = [30, 100]

    # for ws in window_sizes:
    #     for nump in num_of_points:
    #         default_config['window_size'] = ws
    #         default_config['sampling_info']['numpoints'] = nump

    #         mod = EvalStitching(default_config)

    #         time_start = time.time()
    #         results = mod.run()
    #         time_end = time.time()
    #         duration = time_end-time_start
    #         print(f"Time: {duration}")

    #         excel_results = [{
    #             "WindowSize": ws,
    #             "NumPoints": nump,
    #             "Runtime": duration,
    #             "Datatype": 'int16',
    #             "PointSampling": default_config['sampling_info']['sampling_type'],
    #             "Mask Threshold": 5,
    #             "Downsampling Factor": '2**2',
    #         }]

    #         for n in default_config['metrics']:
    #             excel_results[0][n] = np.mean(results[n]["point_metric"])

    #         # "SSIM": np.mean(results['ssim']["point_metric"]),
    #         #     "NCC": np.mean(results['ncc']["point_metric"]),
    #         #     "MI": np.mean(results['nmi']["point_metric"]),
    #         #     "NMI": np.mean(results['issm']["point_metric"]),

    #         file_path = '/scratch/results.xlsx'
    #         create_excel_file(file_path)
    #         append_results_to_excel(file_path, excel_results)

    client.close()



if __name__ == "__main__":
    main()

