from main_qa import (EvalStitching,get_default_config)
import time
from aind_registration_evaluation.util.file_conversion import n5_to_tiff, n5_to_tiff_warped, apply_mask, apply_mask_and_crop, downsample_and_int16, prepareFiles, normalize_image
from aind_registration_evaluation.util.excelOutput import create_excel_file, append_results_to_excel
import os
import numpy as np
import pandas as pd
import multiprocessing
from dask.distributed import Client, LocalCluster


#check ReadMe for options
config_dict = {
    "metrics": ["ncc", "mi", "nmi"],
    "transform_matrix": [
        [1, 0, 0, 0],  # Z
        [0, 1, 0, 0],  # Y
        [0, 0, 1, 0],  # X
        [0, 0, 0, 1]
    ],
}

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

    default_config = config_dict

    # confocal = "/data/coregistration_test_693124_06-21-2024/confocal/693124_confocal_0_300_20xWater_stitched.n5"
    # warpedCortical = "/data/coreg_test_693124_cortical_z_stack_to_0-300.n5"
    # prepareFiles(confocal, warpedCortical, 1, 1)


    default_config["image_1"] = '/scratch/confocal_s0_cropped_dsLev1.tif'
    default_config["image_2"] = '/scratch/cortical_s0_cropped_dsLev1.tif'

    mod = EvalStitching(default_config)
    time_start = time.time()
    results = mod.run_roi(0.008, 0.05, 15, 0.3)
    time_end = time.time()
    duration = time_end-time_start
    print(f"Time: {duration}")



    client.close()



if __name__ == "__main__":
    main()

