import nibabel as nib
from lib.fill_holes import fill_holes
from lib.dilate import dilate
from lib.cylinder import cylinder
from lib.binarize import binarize


def crop(input_nifti, mask, shape="original", label=1, f_dim=3, f_dilations=3, d_iterations=3, d_filling=True, d_dilations=3, c_dilations=3):
    """
    Takes an image and its segmentation, binarizes it for a specific label,
    shapes the binary mask according to the "shape" parameter, and performs
    the multiplication between the image and the mask to keep just the image
    values inside the segmentation area, while all the others are set to zero.
    
    Args:
        input_nifti: nib
            imput object to be shaped
        mask: nib
            segmentation object selecting the area to crop
        shape: choices: {'original', 'fill_holes', 'dilation', 'cylinder'}
            selection of the shape for the image to be cropped. Default is
            'original'
        label: float or int
            label of the voxels to be part of the final mask. Default is 1
        f_dim: int
            dimensions of the squared/cubic structuring element in pixel.
            It is used only if the selected shape is 'fill_holes'.
            Default is 3.
        f_dilations: int
            number of iterations for the dilations before the filling 
            operation, and of erosions after. It is used only if the selected
            shape is 'fill_holes'. Default is 3
        d_iterations: int
            number of iterations for the dilation. It is used only if the
            selected shape is 'dilation'. Default is 3
        d_filling: bool
            if False, only the dilation is performed. If True before the 
            dilation a binary filling operation is performed. It is used 
            only if the selected shape is 'dilation'. Default is True
        d_dilations:: int
            number of iterations for the dilations and then erosions inside
            the filling operation. It is used only if the selected shape is
            'dilation'. Default is 3
        c_dilations: int
            number of dilations to perform before the cylinder shaping to the
            original mask. It is used only if the selected shape is 'dilation'.
            Default is 3

    Returns:
        crop_image: nib
            input image shaped conserving just the information in the area 
            defined by the binary mask
        segm_aligned:
            binary mask shaped
    """

    # mask binarization
    mask = binarize(mask, label=label)

    # Apply shape function on segmentation
    if shape == "fill_holes":
        print(shape)
        mask = fill_holes(mask, dim=f_dim, n_dilations=f_dilations)
    elif shape == "dilate":
        print(shape)
        mask = dilate(mask, iterations=d_iterations, fill=d_filling, n_dilations=d_dilations)
    elif shape == "cylinder":
        print(shape)
        mask = cylinder(mask, c_dilations)
    elif shape == "original":
        print(shape)
    else:
        print("Shape invalid. Going with the original shape.")

    # Put the segmentation into a numpy array
    segmentation = binarized_mask.get_fdata()

    # Put the image into a numpy array
    image = input_nifti.get_fdata()

    # Cut the PET image
    cut_image = image * segmentation
    print(f"done cutting")

    # Save cut image in a NIfTI file
    crop_image = nib.Nifti1Image(cut_image, input_nifti.affine, input_nifti.header)
    segm = nib.Nifti1Image(segmentation, input_nifti.affine, input_nifti.header)

    return crop_image, segm