import numpy as np
from skimage import io, filters, morphology, exposure, measure
import matplotlib.pyplot as plt
import tifffile as tiff
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from scipy.stats import entropy

def get_props(image, binaryThreshold):
    '''
    	- Applies Gaussian blur to smooth the image
		- Creates a binary mask on the image to extract the bright areas. Binary threshold should be adjusted for different images
		- Opens the image to identify single cells more distinct
		- Removes small object that could be rest noise of bright areas in the image which are not a cell
        - Labels the image and extracts the region properties

        Parameters
        ------------------------
        image: 3d numpy array

        binaryThreshold: threshold to create a binary image based on intensity (int)

        Returns
        ------------------------
        regionprops list (see https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops)
    '''
    blurred_image = filters.gaussian(image, sigma=1)
    binary_image = blurred_image > binaryThreshold
    opened_image = morphology.opening(binary_image, morphology.ball(1))
    largeObjects = morphology.remove_small_objects(opened_image, 500)
    labels = morphology.label(largeObjects)
    regions = measure.regionprops(labels)
    return regions

def get_centroid_distances(props1, props2, distanceThreshold):
    '''
        - Calculates distances of the centroids of each cell in one image to each cell in the other image. 
        - A distance threshold is applied to only store the centroids that are close to each other, indicating that these identify the same cells in registered images.
        - Centroids from img1 that all match to the same centroid in img2 but with longer distance are removed because each centroid should have a distinct match (the closest) with a centroid in the other image.

        Parameters
        ------------------------
        props1, props2: regionprops list from both images (see https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops)
        
        distanceThreshold: pixel count for allowed distance of centroids to be seen as a match

        Returns
        ------------------------
        Dictionary of shape {distance: {'Label1': p1.label, 'Centroid1': p1.centroid, 'Bbox1': p1.bbox,
                                    'Label2': p2.label, 'Centroid2': p2.centroid, 'Bbox2': p2.bbox,
                                    'addedDist': addedDist}}
    '''
    distances = {}
    for p1 in props1:
        for p2 in props2:
            dist = np.linalg.norm(np.array(p1.centroid) - np.array(p2.centroid))
            if dist < distanceThreshold:
                addedDist = 0
                while dist in distances:
                    dist += 0.0001
                    addedDist += 0.0001
                distances[dist] = {'Label1': p1.label, 'Centroid1': p1.centroid, 'Bbox1': p1.bbox,
                                    'Label2': p2.label, 'Centroid2': p2.centroid, 'Bbox2': p2.bbox,
                                    'addedDist': addedDist}
    sortedDistances = {k: v for k, v in sorted(distances.items(), key=lambda item: int(item[0]) + item[1]['addedDist'])}
    p1Used = set()
    p2Used = set()
    resultDistances = {}
    for k,v in sortedDistances.items():
        if v['Label1'] in p1Used or v['Label2'] in p2Used: continue
        resultDistances[k] = v
        p1Used.add(v['Label1'])
        p2Used.add(v['Label2'])
    return resultDistances

def create_patches(distances, image_shape, overlapThreshold):
    '''
        - Creates a patch around each matched centroids by adding 3 pixels around the max coordinates of the patches (limited by image bounds).
        - Merges the patches that show more overlap than the overlap threshold, to avoid duplicate evaluations on the same area.

        Parameters
        ------------------------
       distances: Dictionary of shape {distance: {'Label1': p1.label, 'Centroid1': p1.centroid, 'Bbox1': p1.bbox,
                                    'Label2': p2.label, 'Centroid2': p2.centroid, 'Bbox2': p2.bbox,
                                    'addedDist': addedDist}}
       
       image_shape: shape of the two images must be the same. Either can be passed in the form Tuple(z,y,x)

       overlapThreshold: volume overlap theshold above which two patches are merged to their max coordinates (float)

        Returns
        ------------------------
        coordinates of the merged_patches (z_min, y_min, x_min, z_max, y_max, x_max)
    '''
    patches = []
    for distance, info in distances.items():
        bbox1 = info['Bbox1']
        bbox2 = info['Bbox2']
        z_min = 0 if min(bbox1[0],bbox2[0])<4 else (min(bbox1[0],bbox2[0])-3) 
        y_min = 0 if min(bbox1[1],bbox2[1])<4 else (min(bbox1[1],bbox2[1])-3)
        x_min = 0 if min(bbox1[2],bbox2[2])<4 else (min(bbox1[2],bbox2[2])-3)
        z_max = image_shape[0] if max(bbox1[3],bbox2[3])>(image_shape[0]-4) else (max(bbox1[3],bbox2[3])+3)
        y_max = image_shape[1] if max(bbox1[4],bbox2[4])>(image_shape[1]-4) else (max(bbox1[4],bbox2[4])+3)
        x_max = image_shape[2] if max(bbox1[5],bbox2[5])>(image_shape[2]-4) else (max(bbox1[5],bbox2[5])+3)
        patches.append((z_min, y_min, x_min, z_max, y_max, x_max))
    
    merged_patches = []
    while patches:
        z_min, y_min, x_min, z_max, y_max, x_max = patches.pop(0)
        merged = False
        for i, (mz_min, my_min, mx_min, mz_max, my_max, mx_max) in enumerate(merged_patches):
            if not (z_max < mz_min or z_min > mz_max or y_max < my_min or y_min > my_max or x_max < mx_min or x_min > mx_max):
                overlap_patch = (max(z_min, mz_min), max(y_min, my_min), max(x_min, mx_min), min(z_max, mz_max), min(y_max, my_max), min(x_max, mx_max))
                max_patch = (min(z_min, mz_min), min(y_min, my_min), min(x_min, mx_min), max(z_max, mz_max), max(y_max, my_max), max(x_max, mx_max))
                volume_overlap = (overlap_patch[3] - overlap_patch[0]) * (overlap_patch[4] - overlap_patch[1]) * (overlap_patch[5] - overlap_patch[2])
                volume_max_patch = (max_patch[3] - max_patch[0]) * (max_patch[4] - max_patch[1]) * (max_patch[5] - max_patch[2])
                if volume_overlap/volume_max_patch > overlapThreshold:
                    merged_patches[i] = max_patch
                    if i<(len(merged_patches)-1):
                        patches += merged_patches[i+1:]
                        merged_patches = merged_patches[:i+1]
                    merged = True
                    break
        if not merged:
            merged_patches.append((z_min, y_min, x_min, z_max, y_max, x_max))
    return merged_patches

def get_ROIs(img1, img2, img1_binaryThreshold, img2_binaryThreshold, maxCentroidDistance, overlapThreshold):
    '''
        Calls the functions above and returns a dictionary of {patch_coordinates: patch_volume}

        Parameters
        ------------------------
        img1, img2: numpy array of both images. Must be 3d of the same shape

        img1_binaryThreshold, img2_binaryThreshold: threshold to create a binary image based on intensity (int)
        
        maxCentroidDistance: pixel count for allowed distance of centroids to be seen as a match
        
        overlapThreshold: volume overlap theshold above which two patches are merged to their max coordinates (float)

        Returns
        ------------------------
        dictionary of {patch_coordinates: patch_volume}
    '''
    img1_props = get_props(img1, img1_binaryThreshold)
    img2_props = get_props(img2, img2_binaryThreshold)
    print("ConfocalProps:", len(img1_props), "CorticalProps:", len(img2_props))
    distances = get_centroid_distances(img1_props, img2_props, maxCentroidDistance)
    print("Cell Matches:", len(distances))
    patches = create_patches(distances, img1.shape, overlapThreshold)
    patch_dict = {r: ((r[3]-r[0]) * (r[4]-r[1]) * (r[5]-r[2])) for r in patches}
    return patch_dict

def get_ROIs_cellpose(img1, img2, maxCentroidDistance, overlapThreshold):
    '''
        Calls the functions above and returns a dictionary of {patch_coordinates: patch_volume}

        Parameters
        ------------------------
        img1, img2: receives the labeled masks from cellpose (only 3d and must have same shape)

        maxCentroidDistance: pixel count for allowed distance of centroids to be seen as a match
        
        overlapThreshold: volume overlap theshold above which two patches are merged to their max coordinates (float)

        Returns
        ------------------------
        dictionary of {patch_coordinates: patch_volume}
    '''
    img1_regions = measure.regionprops(img1)
    img2_regions = measure.regionprops(img2)
    print("ConfocalProps:", len(img1_regions), "CorticalProps:", len(img2_regions))
    cp_centroid_dist = get_centroid_distances(img1_regions, img2_regions, maxCentroidDistance)
    print("Cell Matches:", len(cp_centroid_dist))
    cp_patches = create_patches(cp_centroid_dist, img1.shape, 0.3)
    patch_dict = {r: ((r[3]-r[0]) * (r[4]-r[1]) * (r[5]-r[2])) for r in cp_patches}
    return patch_dict



def normalized_mutual_information(patch_1, patch_2
    ) -> float:
        """
        Method to compute the mutual information error metric using numpy.
        Note: Check the used dtype to reach a higher precision in the metric

        See: Normalized Mutual Information of: A normalized entropy
        measure of 3-D medical image alignment,
        Studholme,  jhill & jhawkes (1998).

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1
            and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2
            and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the mutual information error.
        """

        patch_1 = patch_1.flatten()  # .astype(np.float64)
        patch_2 = patch_2.flatten()  # .astype(np.float64)

        # Compute the Mutual Information between the two image pixel distributions
        # using skimage
        return normalized_mutual_info_score(patch_1, patch_2, average_method='geometric')

def normalized_cross_correlation(patch_1, patch_2
    ) -> float:
        """
        Method to compute the normalized cross correlation error
        metric based on ITK snap implementation using numpy.
        See detailed description in
        https://itk.org/Doxygen/html/classitk_1_1CorrelationImageToImageMetricv4.html

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1
            and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2
            and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the normalized
            cross correlation error.
        """

        if patch_1.ndim != 1:
            patch_1 = patch_1.flatten()

        if patch_2.ndim != 1:
            patch_2 = patch_2.flatten()

        if patch_1.shape != patch_2.shape:
            raise ValueError("Images must have the same shape")

        mean_patch_1 = np.mean(patch_1, dtype='int16')
        mean_patch_2 = np.mean(patch_2, dtype='int16')

        # Centering values after calculating mean
        centered_patch_1 = patch_1 - mean_patch_1
        centered_patch_2 = patch_2 - mean_patch_2

        numerator = np.inner(centered_patch_1, centered_patch_2) ** 2

        # Calculating 2-norm over centered patches - None means 2-norm
        norm_patch_1 = np.linalg.norm(centered_patch_1, ord=None) ** 2
        norm_patch_2 = np.linalg.norm(centered_patch_2, ord=None) ** 2

        # Multiplicating norms
        denominator = norm_patch_1 * norm_patch_2

        return (numerator / denominator)

def mutual_information(patch_1, patch_2
    ) -> float:
        """
        Method to compute the mutual information error metric using numpy.
        Note: Check the used dtype to reach a higher precision in the metric

        Parameters
        ------------------------
        patch_1: ArrayLike
            2D/3D patch of extracted from the image 1
            and based on a windowed point.

        patch_2: ArrayLike
            2D/3D patch of extracted from the image 2
            and based on a windowed point.

        Returns
        ------------------------
        float
            Float with the value of the mutual information score.
        """

        # # Compute the Mutual Information between the two image pixel distributions
        # # using skimage
        patch_1 = patch_1.flatten() 
        patch_2 = patch_2.flatten() 
        return mutual_info_score(patch_1, patch_2)