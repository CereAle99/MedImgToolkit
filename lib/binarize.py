import nibabel as nib
import numpy as np


def binarize(input_image, label=1):
    """
    Binarize a nibabel (nib) images. All the grey values in the image which level
    is equal to the label are put to 1, all the others are put to 0. 
    
    Args:
        input_image: nib
            imput object of the image to be binarized
        label: float or int
            label of the voxels that has to be put to 1. Default is 1

    Returns: 
        aligned_input: nib
            input_image binarized

    """
    
    # Load the nibabel object image
    array = input_image.get_fdata()

    # Verifies the image is not empty
    if array.size == 0 or np.all(array == 0):
        raise ValueError("Input file is empty")

    # Binarize the image for the label value
    if label == 0:
        array[array != label] = 10
        array[array == label] = 1
        array[array == 10] = 0
    elif label:
        array[array != label] = 0
        array[array == label] = 1

    # Save the image in nibabel object
    binarized_image = nib.Nifti1Image(array, input_image.affine, input_image.header)

    return binarized_image