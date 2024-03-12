import nibabel as nib


import numpy as np
import os
import pytest


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 


from lib.allign_crop import alignment_crop



@pytest.fixture
def sample_multilabel_segmentation():
    """
    Load a nib segmentation with multiple label numbers.
    """
    sample_file_path = os.path.join('data', 'segmentation.nii.gz')
    return nib.load(sample_file_path)


@pytest.fixture
def sample_medical_image():
    """
    Load a nib image representing a PET image.
    """
    sample_file_path = os.path.join('data', 'PT.nii.gz')
    return nib.load(sample_file_path)


@pytest.fixture
def sample_image_center_50_others():
    """
    Fixture: load a NIfTI 5x5x5 cubic array with 50 in the center and the 
    other voxels made random numbers uniformly distributed between 0 and 10
    """
    data = np.random.randint(0, 10, size=(5, 5, 5), dtype=int)
    data[2, 2, 2] = 50

    nifti_image = nib.Nifti1Image(data, affine=np.eye(4))

    return nifti_image


@pytest.fixture
def sample_image_center_50():
    """
    Fixture: load a NIfTI 5x5x5 cubic array with 50 in the voxel (4,4,4) and the 
    other voxels made of 0s
    """
    data = np.zeros((5, 5, 5), dtype=np.uint8)
    data[3, 3, 3] = 50

    nifti_image = nib.Nifti1Image(data, affine=np.eye(4))

    return nifti_image

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
    cube and the other voxels made of 0s. It has also an offset of 
    (-0.25, -0.25, -0.25)
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
def sample_image_center_pixel_double_50():
    """
    Fixture: load a NIfTI 10x10x10 cubic array with 50 in the 2x2x2 centered
    cube and the other voxels made of 0s. It has also an offset of 
    (-0.25, -0.25, -0.25)
    """
    data = np.zeros((10, 10, 10), dtype=np.uint8)
    data[4:6, 4:6, 4:6] = 50

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


def test_alignment_crop_returns_nifti1image(sample_medical_image, sample_multilabel_segmentation):
    """
    Giving to the alignment_crop function a NiftiImage instance

    tests:
    - If the output is a NiftiImage instance
    - If the input image has the same resolution of the output image
    """
    image, mask = alignment_crop(sample_medical_image, sample_multilabel_segmentation)
    image_array = image.get_fdata()
    mask_array = mask.get_fdata()

    assert isinstance(image, nib.Nifti1Image)
    assert isinstance(mask, nib.Nifti1Image)
    assert image_array.shape == mask_array.shape


def test_align_crop_invalid_shape(sample_medical_image, sample_multilabel_segmentation):
    """
    Gets as input an image and a multilabel segmentation, and the invalid 
    shape "wrong" is passed.

    tests:
    - If the function raises a ValueError with the message
    "Shape invalid."
    """

    with pytest.raises(ValueError, match="Shape invalid."):
        alignment_crop(sample_medical_image, sample_multilabel_segmentation, 'wrong')


def test_align_crop_with_shift(sample_image_center_50_others, 
                             sample_image_segm_to_align,
                             sample_image_center_50
                             ):
    """
    Giving a NIfTI 5x5x5 cubic array with 50 in the center and the 
    other voxels made random numbers uniformly distributed between 0 and 10
    and a NIfTI 5x5x5 cubic array with an uncentered voxel with of 1
    and the other voxels made of 0s with offset set to the point 
    (-1,-1,-1) to the alignment_crop function

    tests:
    - If the output image is an image with a 2x2x2 cube of 50 uncentered 
    in th esame position of the second functiona passed
    """

    image, _ = alignment_crop(sample_image_center_50_others, sample_image_segm_to_align, 'original', label=1)
    image = image.get_fdata()
   
    assert np.all(np.round(image) == sample_image_center_50.get_fdata())


def test_align_crop_with_zoom(sample_image_center_50_others, 
                             sample_image_center_pixel_double,
                             sample_image_center_pixel_double_50
                             ):
    """
    Giving a NIfTI 5x5x5 cubic array with 50 in the center and the 
    other voxels made random numbers uniformly distributed between 0 and 10
    and a NIfTI 10x10x10 cubic array with 1 in the 2x2x2 centered
    cube and the other voxels made of 0s with an offset of 
    (-0.25, -0.25, -0.25) to the alignment_crop function

    tests:
    - If the output image is a NIfTI 10x10x10 cubic array with 50 in the 
    2x2x2 centered cube and the other voxels made of 0s.
    """

    image, segm = alignment_crop(sample_image_center_50_others, sample_image_center_pixel_double, 'original', label=1)
    image = image.get_fdata()
    segm = segm.get_fdata()
    image[image>10] = 50
    
    assert np.all(np.round(image) == sample_image_center_pixel_double_50.get_fdata())