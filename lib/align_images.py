import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, shift


def align_images(input_image, reference_image):
    """
    Aligns two nibabel (nib) images with different resolution and pixel-size but spatially registered. 
    The function keep the pixel-size of the reference_image and change the resoluition into 
    a common resolution between the two images in order to align the images' voxels. 
    
    Args:
        input_image: nib
            imput object of the image to be aligned
        reference_image: nib
            imput object of the image taken for reference

    Returns: 
        aligned_input: nib
            input_image resized and aligned with the reference_image
        aligned_reference: nib
            input_image resized and aligned with the reference_image

    """

    # Load the nibabel objects header and image array
    img_header = input_image.header
    ref_header = reference_image.header
    img_affine = input_image.affine
    ref_affine = reference_image.affine
    img_array = input_image.get_fdata()
    ref_array = reference_image.get_fdata()

    # input image resize ratio
    resize_ratio = img_header['pixdim'][1:4] / ref_header['pixdim'][1:4]

    # input image resizing and his displacement
    img_array = zoom(img_array, zoom=resize_ratio, grid_mode=True, mode="grid-constant")
    rest = np.array(img_array.shape) - np.array(img_header['dim'][1:4]) * np.array(resize_ratio)
    pixel_displacement = rest / np.array(img_array.shape)

    # reference image resizing
    ref_array_resized = np.zeros(shape=img_array.shape)
    side_x = (img_array.shape[0] - ref_array.shape[0]) // 2
    side_y = (img_array.shape[1] - ref_array.shape[1]) // 2
    side_z = (img_array.shape[2] - ref_array.shape[2]) // 2
    center_x = ref_array.shape[0]
    center_y = ref_array.shape[1]
    center_z = ref_array.shape[2]
    ref_array_resized[side_x:side_x + center_x, side_y:side_y + center_y, side_z:side_z + center_z] = ref_array

    # input mage axis orientations
    x_orientation = np.sign(img_header['srow_x'][0])
    y_orientation = np.sign(img_header['srow_y'][1])
    z_orientation = np.sign(img_header['srow_z'][2])

    # Managing the offset
    img_header['qoffset_x'] = (img_header['qoffset_x']
                               - x_orientation * (img_header['pixdim'][1] / 2)
                               + x_orientation * (img_header['pixdim'][1] / resize_ratio[0]) / 2)
    img_header['qoffset_y'] = (img_header['qoffset_y']
                               - y_orientation * (img_header['pixdim'][2] / 2)
                               + y_orientation * (img_header['pixdim'][2] / resize_ratio[1]) / 2)
    img_header['qoffset_z'] = (img_header['qoffset_z']
                               - z_orientation * (img_header['pixdim'][3] / 2)
                               + z_orientation * (img_header['pixdim'][3] / resize_ratio[2]) / 2)

    # input image header fixing
    img_header['dim'][1:4] = img_array.shape
    img_affine[0, 0] = x_orientation * (img_header['pixdim'][1] / resize_ratio[0] - pixel_displacement[0])
    img_affine[1, 1] = y_orientation * (img_header['pixdim'][2] / resize_ratio[1] - pixel_displacement[1])
    img_affine[2, 2] = z_orientation * (img_header['pixdim'][3] / resize_ratio[2] - pixel_displacement[2])
    img_affine[0, 3] = img_header['qoffset_x']
    img_affine[1, 3] = img_header['qoffset_y']
    img_affine[2, 3] = img_header['qoffset_z']

    # reference image header fixing
    ref_header['dim'][1:4] = img_array.shape
    ref_affine[0, 0] = x_orientation * (ref_header['pixdim'][1] - pixel_displacement[0])
    ref_affine[1, 1] = y_orientation * (ref_header['pixdim'][2] - pixel_displacement[1])
    ref_affine[2, 2] = z_orientation * (ref_header['pixdim'][3] - pixel_displacement[2])
    ref_affine[0, 3] = ref_header['qoffset_x'] - side_x * (img_affine[0, 0])
    ref_affine[1, 3] = ref_header['qoffset_y'] - side_y * (img_affine[1, 1])
    ref_affine[2, 3] = ref_header['qoffset_z'] - side_z * (img_affine[2, 2])

    # Evaluate the offset and shift the input image
    axis_directions = np.array([-x_orientation, -y_orientation, -z_orientation])
    shift_vector = (ref_affine[0:3, 3] - img_affine[0:3, 3]) / np.abs(np.diag(img_affine)[0:3]) * axis_directions
    img_array = shift(img_array, shift_vector, mode="constant", cval=0)

    # Fix the input image offset
    img_affine[0, 3] = ref_affine[0, 3]
    img_affine[1, 3] = ref_affine[1, 3]
    img_affine[2, 3] = ref_affine[2, 3]

    # reference image and input image ninanel objects assembled
    aligned_input = nib.Nifti1Image(img_array, img_affine, img_header)
    aligned_reference = nib.Nifti1Image(ref_array_resized, ref_affine, ref_header)
    return aligned_input, aligned_reference