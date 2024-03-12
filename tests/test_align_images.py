import nibabel as nib


import numpy as np
import os
import pytest


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 


from lib.align_images import align_images


@pytest.fixture
def sample_medical_image(tmp_path):
    """
    Fixture: Load a NIfTI PET image with one single label number.
    """
    sample_file_path = os.path.join('data', 'PT.nii.gz')
    return nib.load(sample_file_path)


@pytest.fixture
def sample_singlelabel_segmentation():
    """
    Fixture: Load a NIfTI segmentation with one single label number.
    """
    sample_file_path = os.path.join('data', 'segmentation_singlelabel.nii.gz')
    return nib.load(sample_file_path)


@pytest.fixture
def sample_image_center_pixel():
    """
    Fixture: load a NIfTI 5x5x5 cubic array with 1 in the center and the 
    other voxels made of 0s
    """
    data = np.zeros((5, 5, 5), dtype=np.uint8)
    data[2, 2, 2] = 1 

    nifti_image = nib.Nifti1Image(data, affine=np.eye(4))

    return nifti_image

@pytest.fixture
def sample_image_center_pixel_double():
    """
    Fixture: load a NIfTI 10x10x10 cubic array with 1 in the 2x2x2 centered
    cube and the other voxels made of 0s
    """
    data = np.zeros((10, 10, 10), dtype=np.uint8)
    data[4:6, 4:6, 4:6] = 1 

    affine=np.eye(4)
    for i in range(3):
        affine[i,i] = 0.5
    affine[:3, 3] = [-0.25, -0.25, -0.25]
    nifti_image = nib.Nifti1Image(data, affine=affine)

    return nifti_image

@pytest.fixture
def sample_image_segm_to_align():
    """
    Fixture: load a NIfTI 5x5x5 cubic array with an uncentered voxel with of 1
    and the other voxels made of 0s. The NIfTI offset is set to the point 
    (-1,-1,-1)
    """
    data = np.zeros((5, 5, 5), dtype=np.uint8)
    data[3, 3, 3] = 1 

    affine=np.eye(4)
    affine[:3, 3] = [-1, -1, -1]
    nifti_image = nib.Nifti1Image(data, affine=affine)

    return nifti_image

def test_align_images_returns_nifti1image(sample_medical_image, sample_singlelabel_segmentation):
    """
    Giving to the align_images function a NiftiImage instance

    tests:
    - If the output is a NiftiImage instance
    - If the input image has the same resolution of the output image
    - If the datatype of the output image is made of np.unit8 numbers
    """
    image, mask = align_images(sample_medical_image, sample_singlelabel_segmentation)
    image_array = image.get_fdata()
    mask_array = mask.get_fdata()

    assert isinstance(image, nib.Nifti1Image)
    assert isinstance(mask, nib.Nifti1Image)
    assert mask.get_data_dtype() == np.uint8
    assert image_array.shape == mask_array.shape


def test_align_images_same_offset(sample_medical_image, sample_singlelabel_segmentation):
    """
    Giving a NIfTI PET image and a single_label segmentation to the align_images function

    tests:
    - If the two outputs have the same header's offsets
    - If the two outputs have the same pixdim
    """

    image, mask = align_images(sample_medical_image, sample_singlelabel_segmentation)
    image_header = image.header
    mask_header = mask.header

    assert image_header['qoffset_x'] == mask_header['qoffset_x']
    assert image_header['qoffset_y'] == mask_header['qoffset_y']
    assert image_header['qoffset_z'] == mask_header['qoffset_z']
    assert image_header['pixdim'][1] == mask_header['pixdim'][1]
    assert image_header['pixdim'][2] == mask_header['pixdim'][2]
    assert image_header['pixdim'][3] == mask_header['pixdim'][3]


def test_align_images_expected_alignment(sample_image_center_pixel, sample_image_segm_to_align):
    """
    Giving a two 5x5x5 NIfTIImage instance to the align_images function. 
    The first one has a 1 as central voxel and 0s for the others, while 
    the second one has 1 on the voxel (4,4,4) but is spatially aligned
    to the first one for the offset of the origin of (-1,-1,-1)
    
    tests:
    - if the resulting images are identical, that means that the shifting
    is well performed
    """

    image_nifti, segm_nifti = align_images(sample_image_center_pixel, sample_image_segm_to_align)
    image = image_nifti.get_fdata()
    segm = segm_nifti.get_fdata()
    print(segm)
    print(np.round(image))

    assert np.all(np.round(image) == segm)


def test_align_images_expected_resampling(sample_image_center_pixel, sample_image_center_pixel_double):
    """
    Giving a 5x5x5 NIfTIImage instance and a 6x6x6 NIfTIImage instance to 
    the align_images function. The first one has a 1 as central voxel and 
    0s for the others, while the second one is has 1s on the 2x2x2 cube 
    starting from the origin of the array
    
    tests:
    - if the resulting images are identical, that means that the zooming
    is well performed
    """

    image_nifti, segm_nifti = align_images(sample_image_center_pixel, sample_image_center_pixel_double)
    image = image_nifti.get_fdata()
    segm = segm_nifti.get_fdata()
    print(segm)
    print(np.round(image))

    assert np.all(np.round(image) == segm)
