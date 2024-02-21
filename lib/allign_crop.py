import nibabel as nib
from lib.align_images import align_images
from lib.fill_holes import fill_holes
from lib.dilate import dilate
from lib.cylinder import cylinder
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
        mask = fill_holes(mask, f_dilations, f_dim)
    elif shape == "dilation":
        print(shape)
        mask = dilate(mask, d_dilations, d_filling)
    elif shape == "cylinder":
        print(shape)
        mask = cylinder(mask, c_dilations)
    elif shape == "original":
        print(shape)
    else:
        print("Shape invalid. Going with the original shape.")
    print("done shaping")

    # Make PET image and spine segmentation image compatibles
    resized_pet, resized_mask = align_images(input_nifti, mask)
    print("done resizing")

    # Put the segmentation into a numpy array
    segmentation = resized_mask.get_fdata()

    # Put the image into a numpy array
    image = resized_pet.get_fdata()

    # Cut the PET image
    cut_image = image * segmentation
    print(f"done cutting")

    # Save cut image in a NIfTI file
    crop_image = nib.Nifti1Image(cut_image, input_nifti.affine, input_nifti.header)
    segm_aligned = nib.Nifti1Image(segmentation, input_nifti.affine, input_nifti.header)

    return crop_image, segm_aligned