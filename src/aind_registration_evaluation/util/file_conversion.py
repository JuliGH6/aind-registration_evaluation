import tifffile
import z5py
import numpy as np
import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt



def prepareFiles(confocal, warpedCortical, mask_threshold, downsampling_factor):
    if not os.path.isfile('/scratch/confocal_s0.tif'):
        n5_to_tiff(confocal, '/scratch/confocal_s0.tif', 's0')
    if not os.path.isfile('/scratch/cortical_s0.tif'):
        n5_to_tiff_warped(warpedCortical, '/scratch/cortical_s0.tif')
    if not (os.path.isfile('/scratch/confocal_s0_cropped.tif') or os.path.isfile('/scratch/cortical_s0_cropped.tif')):
        apply_mask_and_crop('/scratch/cortical_s0.tif','/scratch/cortical_s0_cropped.tif', '/scratch/confocal_s0.tif', '/scratch/confocal_s0_cropped.tif', mask_threshold)
    if not (os.path.isfile('/scratch/confocal_s0_cropped_downsampled.tif')):
        downsample_and_int16('/scratch/confocal_s0_cropped.tif','/scratch/confocal_s0_cropped_downsampled.tif', downsampling_factor)
    if not (os.path.isfile('/scratch/cortical_s0_cropped_downsampled.tif')):
        downsample_and_int16('/scratch/cortical_s0_cropped.tif','/scratch/cortical_s0_cropped_downsampled.tif', downsampling_factor)


def tiff_to_n5(tiff_filename, n5_filename, downsampling_factors):

    image_volume = tifffile.imread(tiff_filename)
    shape = image_volume.shape
    # Create an .n5 dataset
    f = z5py.File(n5_filename)
    # Create a group for the multiscale data
    if "c0" not in f:
        group = f.create_group("c0")
    else:
        group = f["c0"]

    # Create datasets for each scale level
    for i, factor in enumerate(downsampling_factors):
        # Calculate downsampled shape
        if "s"+str(i) in group:
            del group["s"+str(i)] 
        ds_data = image_volume[::factor, ::factor, ::factor]
        downsampled_shape = ds_data.shape
        ds = group.create_dataset('s'+str(i), shape=downsampled_shape,dtype=image_volume.dtype, chunks=(128, 128, 128), compression="gzip")
        ds[:] = ds_data
        # Optionally, store additional metadata
        ds.attrs["downsamplingFactors"] = tuple([factor,factor,factor])  # Example: store resolution

    # Close the .n5 file
    f.close()


def n5_to_tiff(n5_filename, output_path, scale_level):
    f = z5py.File(n5_filename)
    image = np.array(f["c0"][scale_level])
    tifffile.imwrite(output_path, image)

def n5_to_tiff_warped( n5_filename, output_path):
    f = z5py.File(n5_filename)
    image = np.array(f["c0"])
    tifffile.imwrite(output_path, image)

def apply_mask(inFilepath, outFilepath, threshhold, newValue):
    data = tifffile.imread(inFilepath)
    mask = data <= threshhold
    data[mask] = newValue
    tifffile.imwrite(outFilepath, data)


def apply_mask_and_crop(inFilepath_cortical, outFilepath_cortical, inFilepath_confocal, outFilepath_confocal, threshold):
    data_cortical = tifffile.imread(inFilepath_cortical)
    data_confocal = tifffile.imread(inFilepath_confocal)
    data_cortical = data_cortical.astype(np.int16)
    data_confocal = data_confocal.astype(np.int16)
    mask = data_cortical <= threshold
    data_cortical[mask] = 0

    nonzeros_cortical = np.nonzero(data_cortical)
    # x_min, x_max = nonzeros[0].min(), nonzeros[0].max()
    # y_min, y_max = nonzeros[1].min(), nonzeros[1].max()
    # z_min, z_max = nonzeros[2].min(), nonzeros[2].max()

    x_min_cortical, x_max_cortical = nonzeros_cortical[0].min(), nonzeros_cortical[0].max()
    y_min_cortical, y_max_cortical = nonzeros_cortical[1].min(), nonzeros_cortical[1].max()
    z_min_cortical, z_max_cortical = nonzeros_cortical[2].min(), nonzeros_cortical[2].max()

    # Ensure the cortical bounds do not exceed the confocal dimensions
    x_min= x_min_cortical if x_min_cortical < data_confocal.shape[0] - 1 else data_confocal.shape[0] - 2
    x_max = min(x_max_cortical, data_confocal.shape[0] - 1)
    y_min= y_min_cortical if y_min_cortical < data_confocal.shape[1] - 1 else data_confocal.shape[1] - 2
    y_max = min(y_max_cortical, data_confocal.shape[1] - 1)
    z_min= z_min_cortical if z_min_cortical < data_confocal.shape[2] - 1 else data_confocal.shape[2] - 2
    z_max = min(z_max_cortical, data_confocal.shape[2] - 1)

    cropped_cortical = data_cortical[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    cropped_confocal = data_confocal[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] 

    tifffile.imwrite(outFilepath_confocal, cropped_confocal)
    tifffile.imwrite(outFilepath_cortical, cropped_cortical)

def downsample_and_int16(tiff_path, output_path, downscale_factor):

    # Load the 3D TIFF image stack
    img_stack = tifffile.imread(tiff_path)


    # Define downscaled dimensions (adjust as needed)
    new_width = img_stack.shape[2] // 2 ** downscale_factor
    new_height = img_stack.shape[1] // 2 ** downscale_factor

    # Initialize an empty list to store resized images
    resized_slices = []

    # Resize each slice (2D image) in the 3D stack
    for slice_idx in range(img_stack.shape[0]):
        # Convert each 2D slice to PIL Image
        slice_img = Image.fromarray(img_stack[slice_idx, :, :])

        # Resize the slice
        resized_slice = slice_img.resize((new_width, new_height), Image.LANCZOS)  # Using Lanczos filter

        # Append resized slice to the list
        resized_slices.append(np.array(resized_slice))

    # Convert the list of resized slices back to a numpy array
    resized_img_stack = np.array(resized_slices).astype(np.int16)

    tifffile.imwrite(output_path, resized_img_stack)

def normalize_image(inFilepath, outFilepath):
    image = tifffile.imread(inFilepath)
    normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    tifffile.imwrite(outFilepath, normalized_image)