import nibabel as nib


import numpy as np
import os
import pytest


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 


from lib.crop import crop



@pytest.fixture
def sample_multilabel_segmentation():
    """
    Fixture: Load a NIfTI segmentation with multiple label numbers.
    """
    sample_file_path = os.path.join('data', 'segmentation.nii.gz')
    return nib.load(sample_file_path)



@pytest.fixture
def sample_medical_image():
    """
    Fixture: Load a random NIfTI CT image with one single label number.
    """

    sample_file_path = os.path.join('data', 'segmentation.nii.gz')
    segm = nib.load(sample_file_path)

    data = np.random.randint(0, 10, size=segm.get_fdata().shape, dtype=int)

    nifti_image = nib.Nifti1Image(data, affine=segm.affine)
    return nifti_image


@pytest.fixture
def sample_image_center_cross():
    """
    Fixture: load a NIfTI 5x5x5 cubic array with a central 3x3x3 cross 
    made of 1s in the center and the other voxels made of 0s.
    """
    data = np.zeros((5, 5, 5), dtype=int)
    data[2, 2, 2] = 1
    data[2, 2, 1] = 1
    data[2, 2, 3] = 1
    data[2, 1, 2] = 1
    data[2, 3, 2] = 1
    data[1, 2, 2] = 1
    data[3, 2, 2] = 1

    nifti_image = nib.Nifti1Image(data, affine=np.eye(4))

    return nifti_image


@pytest.fixture
def sample_image_center_cross_50():
    """
    Fixture: load a NIfTI 5x5x5 cubic array with a central 3x3x3 cross 
    made of 50s in the center and the other voxels made of 0s.
    """
    data = np.zeros((5, 5, 5), dtype=int)
    data[2, 2, 2] = 50
    data[2, 2, 1] = 50
    data[2, 2, 3] = 50
    data[2, 1, 2] = 50
    data[2, 3, 2] = 50
    data[1, 2, 2] = 50
    data[3, 2, 2] = 50

    nifti_image = nib.Nifti1Image(data, affine=np.eye(4))

    return nifti_image

@pytest.fixture
def sample_image_center_pixel():
    """
    Fixture: Load a NIfTI 5x5x5 cubic image with 1 in the center and the 
    other voxels made of 0s
    """
    data = np.zeros((5, 5, 5), dtype=np.uint8)
    data[2, 2, 2] = 1 

    nifti_image = nib.Nifti1Image(data, affine=np.eye(4))

    return nifti_image

@pytest.fixture
def sample_image_center_50_others():
    """
    Fixture: load a NIfTI 5x5x5 cubic array with a central 3x3x3 cross 
    made of 50s in the center and the other voxels made of random numbers
    from 0 to 10
    """
    np.random.seed(42)
    data = np.random.randint(0, 10, size=(5, 5, 5), dtype=int)
    data[2, 2, 2] = 50
    data[2, 2, 1] = 50
    data[2, 2, 3] = 50
    data[2, 1, 2] = 50
    data[2, 3, 2] = 50
    data[1, 2, 2] = 50
    data[3, 2, 2] = 50

    nifti_image = nib.Nifti1Image(data, affine=np.eye(4))

    return nifti_image

def test_crop_returns_nifti1image(sample_medical_image, sample_multilabel_segmentation):
    """
    Giving to the crop function a NiftiImage instance

    tests:
    - If the output is a NiftiImage instance
    - If the input image has the same resolution of the output image
    """
    image, mask = crop(sample_medical_image, sample_multilabel_segmentation, label=15)
    image_array = image.get_fdata()
    mask_array = mask.get_fdata()

    assert isinstance(image, nib.Nifti1Image)
    assert isinstance(mask, nib.Nifti1Image)
    assert image_array.shape == mask_array.shape


def test_crop_invalid_shape(sample_medical_image, sample_multilabel_segmentation):
    """
    Gets as input an image and a multilabel segmentation, and the invalid 
    shape "wrong" is passed.

    tests:
    - If the function raises a ValueError with the message
    "Shape invalid."
    """

    with pytest.raises(ValueError, match="Shape invalid."):
        crop(sample_medical_image, sample_multilabel_segmentation, 'wrong')


def test_crop_the_right_part(sample_image_center_50_others, 
                             sample_image_center_pixel, 
                             sample_image_center_cross,
                             sample_image_center_cross_50
                             ):
    """
    Gets as input an image and a multilabel segmentation, and the invalid 
    shape "wrong" is passed.

    tests:
    - If the function raises a ValueError with the message
    "Shape invalid."
    """

    image, segm = crop(sample_image_center_50_others, sample_image_center_pixel, 'dilate', label=1, d_iterations=1)
    image = image.get_fdata()
    segm = segm.get_fdata()
    print(image)
    print(sample_image_center_cross_50.get_fdata())
   
    assert np.all(segm == sample_image_center_cross.get_fdata())
    assert np.all(image == sample_image_center_cross_50.get_fdata())