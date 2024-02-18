import nibabel as nib
from lib.align_images import pet_compatible_to_ct
from lib.fill_holes import fill_spinal_holes
from lib.dilate import dilate_spine
from lib.cylinder import spine_as_cylinder
from lib.binarize import binarize


def crop_spine_shape(input_nifti, mask, shape="original", segmentation_value=1, f_dilations=3, f_dim=3, d_dilations=3, d_filling=True, c_dilations=3):
    """

    Args:
        input_nifti:
        mask:
        shape:
        segmentation_value:
        f_dilations:
        f_dim:
        d_dilations:
        d_filling:
        c_dilations:

    Returns:

    """

    mask = binarize(mask, segmentation_value)

    # Apply shape function on segmentation
    if shape == "fill_holes":
        print(shape)
        mask = fill_spinal_holes(mask, f_dilations, f_dim)
    elif shape == "dilation":
        print(shape)
        mask = dilate_spine(mask, d_dilations, d_filling)
    elif shape == "cylinder":
        print(shape)
        mask = spine_as_cylinder(mask, c_dilations)
    elif shape == "original":
        print(shape)
    else:
        print("Shape invalid. Going with the original shape.")
    print("done shaping")

    # Make PET image and spine segmentation image compatibles
    resized_pet, resized_mask = pet_compatible_to_ct(input_nifti, mask)
    print("done resizing")

    # Put the segmentation into a numpy array
    segmentation = resized_mask.get_fdata()

    # Put the image into a numpy array
    image = resized_pet.get_fdata()

    # Cut the PET image
    cut_image = image * segmentation
    print(f"done cutting")

    # Save cut image in a NIfTI file
    cut_file = nib.Nifti1Image(cut_image, resized_pet.affine, resized_pet.header)

    return cut_file