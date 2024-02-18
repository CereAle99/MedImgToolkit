import nibabel as nib
from lib.fill_holes import fill_spinal_holes
from lib.dilate import dilate_spine
from lib.cylinder import spine_as_cylinder
from lib.binarize import binarize


def crop_spine_from_ct(input_nifti, mask, shape="original", segmentation_value=1):
    """

    Args:
        input_nifti:
        mask:
        shape:
        segmentation_value:

    Returns:

    """

    # Binarize the mask
    binarized_mask = binarize(mask, segmentation_value)

    # Apply shape function on segmentation
    if shape == "fill_holes":
        print(shape)
        binarized_mask = fill_spinal_holes(binarized_mask, 3, 3)
    elif shape == "dilation":
        print(shape)
        binarized_mask = dilate_spine(binarized_mask, 3, True)
    elif shape == "cylinder":
        print(shape)
        binarized_mask = spine_as_cylinder(binarized_mask, 3)
    elif shape == "original":
        print(shape)
    else:
        print("Shape invalid. Going with the original shape.")
    print("done shaping")

    # Put the segmentation into a numpy array
    segmentation = binarized_mask.get_fdata()

    # Put the image into a numpy array
    image = input_nifti.get_fdata()

    # Cut the PET image
    cut_image = image * segmentation
    print(f"done cutting")

    # Save cut image in a NIfTI file
    cut_file = nib.Nifti1Image(cut_image, input_nifti.affine, input_nifti.header)

    return cut_file