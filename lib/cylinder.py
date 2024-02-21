import nibabel as nib
import numpy as np
from lib.dilate import dilate


def cylinder(input_image, dilations=0):
    """
    Takes a nibabel 3D image representing a binary mask and encloses it in a vertical
    cylinder made of 1 values. Previously a dilation of the original binary mask
    may be performed setting the parameter "dilations".
    Args:
        input_image: nib
            imput object to be shaped as a cylinder
        dilations: int
            number of dilations to perform first to the original mask, 
            default is 0

    Returns: nib
        nibabel object representing the binary mask enclosed in a cilinder

    """

    # if the parameter is set, perform the dilation
    if dilations != 0:
        dilated_image = dilate(input_image, dilations)
    else:
        dilated_image = input_image

    # load the image array
    image = dilated_image.get_fdata()

    # evaluate the spatial ranges of the mask
    [i_max, i_min, j_max, j_min, k_max, k_min] = [0, 999, 0, 999, 0, 999]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                if image[i, j, k] != 0:
                    i_max, i_min = np.maximum(i_max, i), np.minimum(i_min, i)
                    j_max, j_min = np.maximum(j_max, j), np.minimum(j_min, j)
                    k_max, k_min = np.maximum(k_max, k), np.minimum(k_min, k)

    # evaluates the radius and center of the cylinder
    cylinder_center = np.array([(i_max + i_min)*0.5, (j_max + j_min)*0.5])
    cylinder_radius = np.maximum((i_max - i_min)*0.5, (j_max - j_min)*0.5)

    # build the cylinder
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            point = np.array([i, j])
            if np.linalg.norm(cylinder_center - point) < cylinder_radius:
                image[i, j, k_min:k_max+1] = 1

    # Put the image in a nibabel object
    output_image = nib.Nifti1Image(image, input_image.affine, input_image.header)

    return output_image