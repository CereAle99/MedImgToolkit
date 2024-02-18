import nibabel as nib


def binarize(input_nifti, label):

    # Load the nifti file image
    image = input_nifti.get_fdata()

    # Binarize the image for the label value
    if label == 1:
        pass
    elif label:
        image[image != label] = 0
        image[image == label] = 1

    # Save the image in nifti file
    binarized_nifti = nib.Nifti1Image(image, input_nifti.affine, input_nifti.header)

    return binarized_nifti