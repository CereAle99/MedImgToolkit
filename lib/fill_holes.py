import nibabel as nib
from scipy import ndimage
import numpy as np


# Fill holes function
def fill_holes(input_nifti, dim=3, n_dilations=None):
    """
    Takes a nibabel 2D/3D image representing a binary mask and fill the 
    holes which are presented as closed structures in the horizontal plane. 
    The structuring element of the operations is a square/cubic element with
    pixel dimensions given by "dim" parameter. The parameter "n_dilations"
    set the number of times a binary dilation is performed before the 
    binary_fill_holes function. After the filling process a binary erosion
    is performed the same amount of times as the dilation. 

    Args:
        input_image: nib
            imput object to be shaped
        dim: int
            dimensions of the squared/cubic structuring element in pixel.
            Default is 3
        n_dilations: int
            number of iterations for the dilations before the filling 
            operation, and of erosions after. Default is None


    Returns: nib
        nibabel object representing the binary mask after teh shaping

    """

    # load the image array
    image = input_nifti.get_fdata()

    # Verifies the image is not empty
    if image.size == 0 or np.all(image == 0):
        raise ValueError("Input file is empty")

    # building the structuring element
    if dim == 0:
        raise ValueError("Dim 0 for the structuring element")
    kernel = np.zeros((dim, dim, dim), dtype=np.uint8)
    kernel[:, dim // 2, :] = 1


    # filling operations, with dilation and erosion if n_dilations is set
    if n_dilations is None:
        final_image = ndimage.binary_fill_holes(image, structure=kernel).astype(int)
    else:
        image = ndimage.binary_dilation(image, iterations=n_dilations)
        image = ndimage.binary_fill_holes(image, structure=kernel).astype(int)
        final_image = ndimage.binary_erosion(image, iterations=n_dilations)

    # Put the image in a NIfTI file
    final_nifti = nib.Nifti1Image(final_image, input_nifti.affine, input_nifti.header)

    return final_nifti