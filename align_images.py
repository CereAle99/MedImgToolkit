import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, shift


def pet_compatible_to_ct(pet_nifti, ct_nifti):
    """

    Args:
        pet_nifti:
        ct_nifti:

    Returns:

    """

    # Load the nifti files header and image array
    pet_header = pet_nifti.header
    ct_header = ct_nifti.header
    pet_affine = pet_nifti.affine
    ct_affine = ct_nifti.affine
    pet_image = pet_nifti.get_fdata()
    ct_image = ct_nifti.get_fdata()

    # PET resize ratio
    resize_ratio = pet_header['pixdim'][1:4] / ct_header['pixdim'][1:4]

    # PET resizing and his displacement
    pet_image = zoom(pet_image, zoom=resize_ratio, grid_mode=True, mode="grid-constant")
    rest = np.array(pet_image.shape) - np.array(pet_header['dim'][1:4]) * np.array(resize_ratio)
    pixel_displacement = rest / np.array(pet_image.shape)

    # CT image resizing
    ct_image_resized = np.zeros(shape=pet_image.shape)
    side_x = (pet_image.shape[0] - ct_image.shape[0]) // 2
    side_y = (pet_image.shape[1] - ct_image.shape[1]) // 2
    side_z = (pet_image.shape[2] - ct_image.shape[2]) // 2
    center_x = ct_image.shape[0]
    center_y = ct_image.shape[1]
    center_z = ct_image.shape[2]
    ct_image_resized[side_x:side_x + center_x, side_y:side_y + center_y, side_z:side_z + center_z] = ct_image

    # Image axis orientations
    x_orientation = np.sign(pet_header['srow_x'][0])
    y_orientation = np.sign(pet_header['srow_y'][1])
    z_orientation = np.sign(pet_header['srow_z'][2])

    # Managing the offset
    pet_header['qoffset_x'] = (pet_header['qoffset_x']
                               - x_orientation * (pet_header['pixdim'][1] / 2)
                               + x_orientation * (pet_header['pixdim'][1] / resize_ratio[0]) / 2)
    pet_header['qoffset_y'] = (pet_header['qoffset_y']
                               - y_orientation * (pet_header['pixdim'][2] / 2)
                               + y_orientation * (pet_header['pixdim'][2] / resize_ratio[1]) / 2)
    pet_header['qoffset_z'] = (pet_header['qoffset_z']
                               - z_orientation * (pet_header['pixdim'][3] / 2)
                               + z_orientation * (pet_header['pixdim'][3] / resize_ratio[2]) / 2)

    # PET header fixing
    pet_header['dim'][1:4] = pet_image.shape
    pet_affine[0, 0] = x_orientation * (pet_header['pixdim'][1] / resize_ratio[0] - pixel_displacement[0])
    pet_affine[1, 1] = y_orientation * (pet_header['pixdim'][2] / resize_ratio[1] - pixel_displacement[1])
    pet_affine[2, 2] = z_orientation * (pet_header['pixdim'][3] / resize_ratio[2] - pixel_displacement[2])
    pet_affine[0, 3] = pet_header['qoffset_x']
    pet_affine[1, 3] = pet_header['qoffset_y']
    pet_affine[2, 3] = pet_header['qoffset_z']

    # CT header fixing
    ct_header['dim'][1:4] = pet_image.shape
    ct_affine[0, 0] = x_orientation * (ct_header['pixdim'][1] - pixel_displacement[0])
    ct_affine[1, 1] = y_orientation * (ct_header['pixdim'][2] - pixel_displacement[1])
    ct_affine[2, 2] = z_orientation * (ct_header['pixdim'][3] - pixel_displacement[2])
    ct_affine[0, 3] = ct_header['qoffset_x'] - side_x * (pet_affine[0, 0])
    ct_affine[1, 3] = ct_header['qoffset_y'] - side_y * (pet_affine[1, 1])
    ct_affine[2, 3] = ct_header['qoffset_z'] - side_z * (pet_affine[2, 2])

    # Evaluate the offset and shift the PET image
    axis_directions = np.array([-x_orientation, -y_orientation, -z_orientation])
    shift_vector = (ct_affine[0:3, 3] - pet_affine[0:3, 3]) / np.abs(np.diag(pet_affine)[0:3]) * axis_directions
    print(shift_vector)
    pet_image = shift(pet_image, shift_vector, mode="constant", cval=0)

    # Fix the PET offset
    pet_affine[0, 3] = ct_affine[0, 3]
    pet_affine[1, 3] = ct_affine[1, 3]
    pet_affine[2, 3] = ct_affine[2, 3]

    # CT and PET NIfTI files assembled
    resized_pet = nib.Nifti1Image(pet_image, pet_affine, pet_header)
    resized_ct = nib.Nifti1Image(ct_image_resized, ct_affine, ct_header)
    return resized_pet, resized_ct