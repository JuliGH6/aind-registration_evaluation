from ..src.main_qa import (EvalStitching,get_default_config)
import time
from aind_registration_evaluation.util.file_conversion import n5_to_tiff, n5_to_tiff_warped, apply_mask, apply_mask_and_crop, downsample_and_int16, prepareFiles, normalize_image
from aind_registration_evaluation.util.excelOutput import create_excel_file, append_results_to_excel
from aind_registration_evaluation.util.calculateAverage import create_transformation_matrices
import os
import numpy as np
import pandas as pd
import multiprocessing
from dask.distributed import Client, LocalCluster
import tifffile as tiff


#check ReadMe for options
config_dict = {
    "image_channel": 0,
    "data_type": "small",
    "metrics": ["ncc", "mi", "nmi"],
    #"ksam" is not implemented
    "window_size": 10,
    "sampling_info": {
        "sampling_type": "grid",
        "numpoints": 200
    },
    "visualize": False,
}

def main():

 
    default_config = config_dict

    n_workers = multiprocessing.cpu_count()
    threads_per_worker = 1
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=True,
        memory_limit="auto",
    )

    client = Client(cluster)

    confocals = ['/root/capsule/data/693124_coregistration/confocal_s0_cropped_dsLev1.tif']
    corticals = ['/root/capsule/data/693124_coregistration/cortical_s0_cropped_dsLev1.tif']
    
    shape = tiff.imread('/root/capsule/data/693124_coregistration/confocal_s0_cropped_dsLev1.tif').shape

    print(shape)
    matrizes = create_transformation_matrices(shape)
    # for matrix in matrizes:
    #     default_config["transform_matrix"] = matrix
    #     print(matrix)
    #     joined_rows = [",".join([str(value) for value in row]) for row in matrix]
    #     excelMatrix = "|".join(joined_rows) 
    #     for i in range(len(corticals)):
    #         default_config["image_1"] = confocals[i]
    #         default_config["image_2"] = corticals[i]
    #         mod = EvalStitching(default_config)

    #         time_start = time.time()
    #         results = mod.run()
    #         time_end = time.time()
    #         duration = time_end-time_start
    #         print(f"Time: {duration}")

    #         excel_results = [{
    #             "Runtime": duration,
    #             "Downsampling Factor": '2** 2', #+ str(i+1),
    #             "Matrix": excelMatrix
    #         }]

    #         for n in default_config['metrics']:
    #             excel_results[0][n] = np.mean(results[n]["point_metric"])

    #         file_path = '/scratch/results_t.xlsx'
    #         create_excel_file(file_path)
    #         append_results_to_excel(file_path, excel_results)

    if not os.path.exists("/scratch/results_roi_m.xlsx"):
        labels = ["NumberROIs", "MaxCentroidDistance", "Runtime", "Downsampling Factor", "Matrix", "ncc", "mi", "nmi"]
        initial_data = pd.DataFrame(columns=labels)
        initial_data.to_excel("/scratch/results_roi_m.xlsx", index=False)

    maxCentroidDistance = [5]

    roi_dict = {"metrics": ["ncc", "mi", "nmi"]}

    # valuesDict = {}

    for matrix in matrizes:
        roi_dict["transform_matrix"] = matrix
        joined_rows = [",".join([str(value) for value in row]) for row in matrix]
        excelMatrix = "|".join(joined_rows)        
        for i in range(len(corticals)):
            roi_dict["image_1"] = confocals[i]
            roi_dict["image_2"] = corticals[i]
            for mCd in maxCentroidDistance:

                    mod = EvalStitching(roi_dict)

                    time_start = time.time()
                    results = mod.run_roi(0.008, 0.05, mCd, 0.3)
                    time_end = time.time()
                    duration = time_end-time_start
                    print(f"Time: {duration}")

                    excel_results = [{
                        "NumberROIs": results["num_rois"],
                        "MaxCentroidDistance": mCd,
                        "Runtime": duration,
                        "Downsampling Factor": '2**1', #+ str(i+1),
                        "Matrix": excelMatrix
                    }]

                    for n in roi_dict['metrics']:
                        excel_results[0][n] = results[n]["weighted_avg"]
                        # valuesDict[n] = 
            

                    file_path = '/scratch/results_roi_m.xlsx'
                    append_results_to_excel(file_path, excel_results)

    client.close()



if __name__ == "__main__":
    main()

