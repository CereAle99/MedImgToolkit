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
    """
    Fixture: Load a NIfTI segmentation with one single label number.
    """
    sample_file_path = os.path.join('data', 'segmentation_singlelabel.nii.gz')
    return nib.load(sample_file_path)


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
def sample_image_center_big_cross():
    """
    Fixture: load a NIfTI 5x5x5 cubic array with a central 5x5x5 cross 
    made of 1s in the center and the other voxels made of 0s.
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


def test_dilate_returns_nifti1image(sample_singlelabel_segmentation):
    """
    Giving to the dilate function a NiftiImage instance

    tests:
    - If the output is a NiftiImage instance
    - If the input image has the same resolution of the output image
    - If the datatype of the output image is made of np.unit8 numbers
    """
    result = dilate(sample_singlelabel_segmentation)
    result_data = result.get_fdata()
    sample_data = sample_singlelabel_segmentation.get_fdata()

    assert isinstance(result, nib.Nifti1Image)
    assert result_data.shape == sample_data.shape
    assert result.get_data_dtype() == np.uint8


def test_dilate_empty_input():
    """
    Giving an empty segmentation to the dilate function
    tests:
    - If the function raises a ValueError with the message
    "Input file is empty."
    """

    empty_image = nib.Nifti1Image(np.zeros((100, 100, 100)), np.eye(4))

    with pytest.raises(ValueError, match="Input file is empty"):
        dilate(empty_image)


def test_dilate_encloses_mask(sample_singlelabel_segmentation):
    """
    Giving a single-label segmentation to the dilate function

    tests:
    - If the output image encloses the input image, that means that all the 
    voxel having 1s in the input image are 1s also in the output image
    """

    dilate_mask = dilate(sample_singlelabel_segmentation).get_fdata()
    input_mask = sample_singlelabel_segmentation.get_fdata()

    assert np.all(np.logical_or(np.equal(dilate_mask, input_mask), dilate_mask))


def test_dilate_output_differs_from_input(sample_singlelabel_segmentation):
    """
    Giving a single-label segmentation to the dilate function

    tests:
    - If input and output are different images
    """
    input_image = sample_singlelabel_segmentation.get_fdata()
    output_image = dilate(sample_singlelabel_segmentation)
    
    assert not np.array_equal(input_image, output_image)


def test_dilate_iterations_param_limits(sample_singlelabel_segmentation):
    """
    Giving a single-label segmentation to the dilate function and 0 as 
    iteration paramenter
    
    tests:
    - If the function raises a ValueError with the message 
    "Dim 0 for the structuring element."
    """

    with pytest.raises(ValueError, match="Dim 0 for the structuring element"):
        dilate(sample_singlelabel_segmentation, iterations=0)


def test_dilate_iterations_param_limits(sample_singlelabel_segmentation):
    """
    Giving a single-label segmentation to the dilate function, and evaluating
    once with an iteration parameter 1 and once with a parameter 2

    tests:
    - If the outputs of the two cases are different images
    - If the output image with an iteration parameter of 2 encloses the other,
    that means that all the voxel having 1s in the input image are 1s also 
    in the output image 
    """

    dilate_dim_1 = dilate(sample_singlelabel_segmentation, iterations=1).get_fdata()
    dilate_dim_2 = dilate(sample_singlelabel_segmentation, iterations=2).get_fdata()

    assert np.any(dilate_dim_1 != dilate_dim_2)
    assert np.all(np.logical_or(np.equal(dilate_dim_2, dilate_dim_1), dilate_dim_2))


def test_dilate_segmentation_is_bigger(sample_singlelabel_segmentation):
    """
    Givin a single-label segmentation to the dilate function

    tests:
    If the sum of the labels in the output image is greater than in the 
    input image
    """

    input_mask = sample_singlelabel_segmentation.get_fdata()
    dilate_mask = dilate(sample_singlelabel_segmentation, iterations=2).get_fdata()

    assert np.sum(dilate_mask, axis=(0,1,2)) > np.sum(input_mask, axis=(0,1,2))


def test_dilate_dilating_pixel(sample_image_center_pixel, sample_image_center_cross):
    """
    Giving a 5x5x5 single-label segmentation with a 1 in the central voxel and 0s
    in the others to the dilation function

    tests:
    - If the output image represents a 3x3x3 centered cross
    """

    dilate_mask = dilate(sample_image_center_pixel).get_fdata()
    cross_mask = sample_image_center_cross.get_fdata()

    assert np.all(dilate_mask == cross_mask)


def test_dilate_dilating_pixel_2(sample_image_center_pixel, sample_image_center_big_cross):
    """
    Giving a 5x5x5 single-label segmentation with a 1 in the central 
    voxel and 0s in the others to the dilation function

    tests:
    - If the output of the dilation represents a 5x5x5 
    asterisk shaped image
    """

    dilate_mask = dilate(sample_image_center_pixel, iterations=2).get_fdata()
    cross_mask = sample_image_center_big_cross.get_fdata()

    assert np.all(dilate_mask == cross_mask)