import nibabel as nib
from scipy import ndimage
from lib.fill_holes import fill_holes
import numpy as np


def dilate(imput_image, iterations=1, fill=False, dim=3, n_dilations=None):
    """
    Takes a nibabel 2D/3D image representing a binary mask and dilates the mask 
    a number of times given by the "iterations" parameter. If the parameter 
    "fill" is True firstly the function fill_holes is executed on the mask.
    Args:
        input_image: nib
            imput object to be shaped
        iterations: int
            number of iterations for the dilation. Default is 1
        fill: bool
            if False, only the dilation is performed. If True before the 
            dilation a binary filling operation is performed. Default is 
            False
        dim: int
            dimensions of the squared/cubic structuring element in pixel for
            the filling operation. Default is 3
        n_dilations: int
            number of iterations for the dilations and then erosions inside
            the filling operation. Default is None


    Returns: nib
        nibabel object representing the binary mask after teh shaping

    """
    if fill:
        fill_nifti = fill_holes(imput_image, dim=dim, n_dilations=n_dilations)
    else:
        fill_nifti = imput_image

    image = fill_nifti.get_fdata()

    # Verifies the image is not empty
    if image.size == 0 or np.all(image == 0):
        raise ValueError("Input file is empty")

    # Dilate the mask
    final_image = ndimage.binary_dilation(image, iterations=iterations)

    # Put the image in a NIfTI file
    final_nifti = nib.Nifti1Image(final_image, imput_image.affine, imput_image.header)

    return final_nifti