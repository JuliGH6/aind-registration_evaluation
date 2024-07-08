import os
import numpy as np
import pandas as pd

def create_excel_summary(file_path, duration, transformation_matrix, results, metric_names, window_size, numpoints):

    data = {
        'Metric': [],
        'Mean': [],
        'Std': []
    }

    for metric_name in metric_names:
        mean_value = np.mean(results[metric_name]["point_metric"])
        std_value = np.std(results[metric_name]["point_metric"])
        data['Metric'].append(metric_name)
        data['Mean'].append(mean_value)
        data['Std'].append(std_value)

    df_metrics = pd.DataFrame(data)

    metrics_rows = len(metric_names) + 1  # +1 for header row
    
    # Create DataFrame for computation time
    df_computation_time = pd.DataFrame({'Computation Time': [duration]})
    
    # Create DataFrame for transformation matrix
    df_transformation = pd.DataFrame(transformation_matrix)

    df_window_size = pd.DataFrame({'window size': [window_size]})

    df_numpoints = pd.DataFrame({'number of points': [numpoints]})
    
    # Write to Excel file
    with pd.ExcelWriter(file_path) as writer:
        # Write metrics data
        df_metrics.to_excel(writer, sheet_name='Summary', startrow=0, startcol=0, index=False)
        
        current_row = metrics_rows + 3  # Start computation time 3 rows after metrics table
        df_computation_time.to_excel(writer, sheet_name='Summary', startrow=current_row, startcol=0, index=False, header=None)

        df_window_size.to_excel(writer, sheet_name='Summary', startrow=0, startcol=5, index=False, header=None)
        df_numpoints.to_excel(writer, sheet_name='Summary', startrow=1, startcol=5, index=False, header=None)
        
        # Add space between computation time and transformation matrix (3 blank rows)
        current_row += 3
        df_transformation.to_excel(writer, sheet_name='Summary', startrow=current_row, startcol=0, index=False, header=None)

def create_excel_file(file_path):
    if not os.path.exists(file_path):
        labels = ["WindowSize", "NumPoints", "Runtime", "Datatype", "PointSampling", "Mask Threshold", "Downsampling Factor", "ssim", "ncc", "mi", "nmi"]
        initial_data = pd.DataFrame(columns=labels)
        initial_data.to_excel(file_path, index=False)

def append_results_to_excel(file_path, results):
    # Read the existing data
    data = pd.read_excel(file_path)
    
    # Convert results to a DataFrame
    new_data = pd.DataFrame(results)
    
    # Append new results to the existing data
    updated_data = pd.concat([data, new_data], ignore_index=True)
    
    # Save the updated data back to the Excel file
    updated_data.to_excel(file_path, index=False)
