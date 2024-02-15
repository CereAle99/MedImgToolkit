import nibabel as nib
from scipy import ndimage
import numpy as np


# Fill holes function
def fill_spinal_holes(input_nifti, n_dilations=3, dim=3):
    """

    Args:
        input_nifti:
        dim:
        n_dilations:

    Returns:

    """

    image = input_nifti.get_fdata()
    kernel = np.zeros((dim, dim, dim), dtype=np.uint8)
    kernel[:, dim // 2, :] = 1

    image = ndimage.binary_dilation(image, iterations=n_dilations)
    image = ndimage.binary_fill_holes(image, structure=kernel).astype(int)
    final_image = ndimage.binary_erosion(image, iterations=n_dilations)

    # Put the image in a NIfTI file
    final_nifti = nib.Nifti1Image(final_image, input_nifti.affine, input_nifti.header)

    return final_nifti