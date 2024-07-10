import os
import numpy as np
import pandas as pd

def create_excel_file(file_path):
    '''
    If file_path doesn't exist already, it creates an excel file with the columns predetermined to be 
    "WindowSize", "NumPoints", "Runtime", "Datatype", "PointSampling", "Downsampling Factor", "Matrix", "ncc", "mi", "nmi"
    You can change that by changing labels.

    Parameters
        ------------------------
        file_path: new excel output path

        Returns
        ------------------------
        none

    '''
    if not os.path.exists(file_path):
        labels = ["WindowSize", "NumPoints", "Runtime", "Datatype", "PointSampling", "Downsampling Factor", "Matrix", "ncc", "mi", "nmi"]
        initial_data = pd.DataFrame(columns=labels)
        initial_data.to_excel(file_path, index=False)

def append_results_to_excel(file_path, results):
    '''
    It writes all results into the excel file specified in file_path

    Parameters
        ------------------------
        file_path: new excel output path (string)
        results: a list with a dictionary inside with the values that should be written to the excel file

        Returns
        ------------------------
        none

    '''
    # Read the existing data
    data = pd.read_excel(file_path)
    
    # Convert results to a DataFrame
    new_data = pd.DataFrame(results)
    
    # Append new results to the existing data
    updated_data = pd.concat([data, new_data], ignore_index=True)
    
    # Save the updated data back to the Excel file
    updated_data.to_excel(file_path, index=False)
