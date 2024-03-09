import nibabel as nib


import numpy as np
import os
import pytest


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 


from lib.dilate import dilate

@pytest.fixture
def sample_singlelabel_segmentation():
    sample_file_path = os.path.join('data', 'segmentation_singlelabel.nii.gz')
    return nib.load(sample_file_path)


@pytest.fixture
def sample_image_center_pixel():
    """
    Create a 5x5x5 cube with a center made of 1s with borders made of 0s
    Put it in a NIfTI image.
    """
    data = np.zeros((5, 5, 5), dtype=np.uint8)
    data[2, 2, 2] = 1 

    nifti_image = nib.Nifti1Image(data, affine=np.eye(4))

    return nifti_image


@pytest.fixture
def sample_image_center_big_cross():
    """
    Create a 9x9x9 cube with a Central cubic core with removed 
    corners and diagonal and vertical bars. Put it in a NIfTI image.
    """
    data = np.zeros((5, 5, 5), dtype=int)
    data[1:4, 1:4, 1:4] = 1
    data[1, 1, 1] = 0
    data[1, 1, 3] = 0
    data[1, 3, 1] = 0
    data[1, 3, 3] = 0
    data[3, 1, 1] = 0
    data[3, 1, 3] = 0
    data[3, 3, 1] = 0
    data[3, 3, 3] = 0
    data[2, 2, 0] = 1
    data[0, 2, 2] = 1
    data[2, 0, 2] = 1
    data[2, 4, 2] = 1
    data[4, 2, 2] = 1
    data[2, 2, 4] = 1

    nifti_image = nib.Nifti1Image(data, affine=np.eye(4))

    return nifti_image



@pytest.fixture
def sample_image_center_cross():
    """
    Create a 9x9x9 cube with a center made of 1s with borders made of 0s
    Put it in a NIfTI image.
    """
    data = np.zeros((5, 5, 5), dtype=int)
    data[2, 2, 2] = 1.
    data[2, 2, 1] = 1.
    data[2, 2, 3] = 1.
    data[2, 1, 2] = 1.
    data[2, 3, 2] = 1.
    data[1, 2, 2] = 1.
    data[3, 2, 2] = 1.

    nifti_image = nib.Nifti1Image(data, affine=np.eye(4))

    return nifti_image


def test_dilate_returns_nifti1image(sample_singlelabel_segmentation):
    """
    Tests:
    If the output is a NiftiImage instance
    If the input image has the same resolution of the output image
    If the datatype of the output image is binary
    """
    result = dilate(sample_singlelabel_segmentation)
    result_data = result.get_fdata()
    sample_data = sample_singlelabel_segmentation.get_fdata()

    assert isinstance(result, nib.Nifti1Image)
    assert result_data.shape == sample_data.shape
    assert result.get_data_dtype() == np.uint8


def test_dilate_empty_input():
    """
    Tests:
    If the function gets an empty input
    """

    empty_image = nib.Nifti1Image(np.zeros((100, 100, 100)), np.eye(4))

    with pytest.raises(ValueError, match="Input file is empty"):
        dilate(empty_image)


def test_dilate_encloses_mask(sample_singlelabel_segmentation):
    """
    Tests:
    If the output encloses the input image
    """

    dilate_mask = dilate(sample_singlelabel_segmentation).get_fdata()
    input_mask = sample_singlelabel_segmentation.get_fdata()

    assert np.all(np.logical_or(np.equal(dilate_mask, input_mask), dilate_mask))


def test_dilate_output_differs_from_input(sample_singlelabel_segmentation):
    """
    Tests:
    If input and output are different
    """
    input_image = sample_singlelabel_segmentation.get_fdata()
    output_image = dilate(sample_singlelabel_segmentation)
    
    assert not np.array_equal(input_image, output_image)


def test_dilate_iterations_param_limits(sample_singlelabel_segmentation):
    """
    Tests:
    If giving 0 as structuring element raises an error
    """

    with pytest.raises(ValueError, match="Dim 0 for the structuring element"):
        dilate(sample_singlelabel_segmentation, iterations=0)


def test_dilate_iterations_param_limits(sample_singlelabel_segmentation):
    """
    Tests:
    If the output for a higher interation parameter encloses the one with a lower one
    """

    dilate_dim_1 = dilate(sample_singlelabel_segmentation, iterations=1).get_fdata()
    dilate_dim_2 = dilate(sample_singlelabel_segmentation, iterations=2).get_fdata()

    assert np.all(np.logical_or(np.equal(dilate_dim_2, dilate_dim_1), dilate_dim_2))


def test_dilate_segmentation_is_bigger(sample_singlelabel_segmentation):
    """
    Tests:
    If there are more labels in the output image than in the input
    """

    input_mask = sample_singlelabel_segmentation.get_fdata()
    dilate_mask = dilate(sample_singlelabel_segmentation, iterations=2).get_fdata()

    assert np.sum(dilate_mask, axis=(0,1,2)) > np.sum(input_mask, axis=(0,1,2))


def test_dilate_dilating_pixel(sample_image_center_pixel, sample_image_center_cross):
    """
    The dilate function get as input a 5x5x5 nibabel image with just a 
    certral pixel.

    Tests:
    If the output of the dilation is an identical image with a 3x3x3
    centered cross.
    """

    dilate_mask = dilate(sample_image_center_pixel).get_fdata()
    cross_mask = sample_image_center_cross.get_fdata()

    assert np.all(dilate_mask == cross_mask)


def test_dilate_dilating_pixel_2(sample_image_center_pixel, sample_image_center_big_cross):
    """
    The dilate function get as input a 5x5x5 nibabel image with just a 
    certral pixel.

    Tests:
    If the output of the dilation is an identical image with a 5x5x5 
    central cubic core with removed corners and diagonal and vertical bars.
    """

    dilate_mask = dilate(sample_image_center_pixel, iterations=2).get_fdata()
    cross_mask = sample_image_center_big_cross.get_fdata()

    assert np.all(dilate_mask == cross_mask)