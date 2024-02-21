import nibabel as nib


def binarize(input_image, label):
    """
    Binarize a nibabel (nib) images. All the grey values in the image which level
    is equal to the label are put to 1, all the others are put to 0. 
    
    Args:
        input_image: nib
            imput object of the image to be binarized
        label: float or int
            label of the voxels that has to be put to 1

    Returns: 
        aligned_input: nib
            input_image binarized

    """
    
    # Load the nibabel object image
    array = input_image.get_fdata()

    # Binarize the image for the label value
    if label == 1:
        pass
    elif label:
        array[array != label] = 0
        array[array == label] = 1

    # Save the image in nibabel object
    binarized_image = nib.Nifti1Image(array, input_image.affine, input_image.header)

    return binarized_image