import nibabel as nib
from scipy import ndimage
from lib.fill_holes import fill_spinal_hole


def dilate_spine(input_nifti, iterations=3, fill=False):
    """

    Args:
        input_nifti:
        iterations:
        fill:

    Returns:

    """
    if fill:
        fill_nifti = fill_spinal_holes(input_nifti, 3)
    else:
        fill_nifti = input_nifti

    image = fill_nifti.get_fdata()
    final_image = ndimage.binary_dilation(image, iterations=iterations)

    # Put the image in a NIfTI file
    final_nifti = nib.Nifti1Image(final_image, input_nifti.affine, input_nifti.header)

    return final_nifti