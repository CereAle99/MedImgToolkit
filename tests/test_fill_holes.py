import nibabel as nib
import numpy as np
import scipy as sp
import os
import pytest


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 


from lib.fill_holes import fill_holes

@pytest.fixture
def sample_singlelabel_segmentation():
    """
    Fixture: Load a NIfTI segmentation with one single label number.
    """
    sample_file_path = os.path.join('data', 'segmentation_singlelabel.nii.gz')
    return nib.load(sample_file_path)

@pytest.fixture
def sample_image_with_hole():
    """
    Fixture: Load a NIfTI 5x5x5 cube with a hole in the center and borders made of 0s
    """
    data = np.zeros((5, 5, 5), dtype=np.uint8)
    data[1:4, 1:4, 1:4] = 1 
    data[2:3, 2:3, 2:3] = 0

    nifti_image = nib.Nifti1Image(data, affine=np.eye(4))

    return nifti_image

@pytest.fixture
def sample_image_without_hole():
    """
    Fixture: Load a NIfTI 5x5x5 cube with a center made of 1s with borders made of 0s
    """
    data = np.zeros((5, 5, 5), dtype=np.uint8)
    data[1:4, 1:4, 1:4] = 1 

    nifti_image = nib.Nifti1Image(data, affine=np.eye(4))

    return nifti_image


@pytest.fixture
def sample_image_with_hole_9():
    """
    Fixture: Load a NIfTI 9x9x9 cube with a 5x5x5 hole in the center and borders made of 0s
    """
    data = np.zeros((9, 9, 9), dtype=np.uint8)
    data[1:8, 1:8, 1:8] = 1 
    data[2:7, 2:7, 2:7] = 0

    nifti_image = nib.Nifti1Image(data, affine=np.eye(4))

    return nifti_image

@pytest.fixture
def sample_image_without_hole_9():
    """
    Fixture: Load a NIfTI 9x9x9 cube with a centered 7x7x7 cube of 1s with borders made of 0s
    """
    data = np.zeros((9, 9, 9), dtype=np.uint8)
    data[1:8, 1:8, 1:8] = 1 

    nifti_image = nib.Nifti1Image(data, affine=np.eye(4))

    return nifti_image





def test_fill_holes_returns_nifti1image(sample_singlelabel_segmentation):
    """
    Giving a single-label segmentation to the fill_holes function

    tests:
    - If the output is a NiftiImage instance
    - If the input image has the same resolution of the output image
    - If the datatype of the output image is made of np.unit8 numbers
    """
    result = fill_holes(sample_singlelabel_segmentation)
    result_array = result.get_fdata()
    sample_array = sample_singlelabel_segmentation.get_fdata()

    assert isinstance(result, nib.Nifti1Image)
    assert result_array.shape == sample_array.shape
    assert result.get_data_dtype() == np.uint8


def test_fill_holes_empty_input():
    """
    Giving an empty segmentation to the fill_holes function
    tests:
    - If the function raises a ValueError with the message
    "Input file is empty."
    """

    empty_image = nib.Nifti1Image(np.zeros((100, 100, 100)), np.eye(4))

    with pytest.raises(ValueError, match="Input file is empty"):
        fill_holes(empty_image)


def test_fill_holes_encloses_mask(sample_singlelabel_segmentation):
    """
    Giving a single-label segmentation to the fill_holes function

    tests:
    - If the output image encloses the input image, that means that all the 
    voxel having 1s in the input image are 1s also in the output image
    """

    fill_holes_mask = fill_holes(sample_singlelabel_segmentation).get_fdata()
    input_mask = sample_singlelabel_segmentation.get_fdata()

    assert np.all(np.logical_or(np.equal(fill_holes_mask, input_mask), fill_holes_mask))


def test_fill_holes_output_differs_from_input(sample_singlelabel_segmentation):
    """
    Giving a single-label segmentation to the fill_holes function

    tests:
    - If input and output are different images
    """
    input_image = sample_singlelabel_segmentation.get_fdata()
    output_image = fill_holes(sample_singlelabel_segmentation)
    
    assert not np.array_equal(input_image, output_image)


def test_fill_holes_dim_param_limits(sample_singlelabel_segmentation):
    """
    Giving a segmentation to the fill_holes function

    tests:
    - If Giving 0 as structuring element raises an error
    """

    with pytest.raises(ValueError, match="Dim 0 for the structuring element"):
        fill_holes(sample_singlelabel_segmentation, dim=0)


def test_fill_holes_dim_param_hole_mgnitude(sample_singlelabel_segmentation):
    """
    Giving a single-label segmentation to the fill_holes function, and calling 
    the function once with a dim parameter of 1 and once with a dim parameter of 2

    tests:
    - If for the output of the function with the dim parameter of 2 the number of
    0 labels is higher than for the output with a dim parameter of 1 
    """

    fill_holes_dim_1 = fill_holes(sample_singlelabel_segmentation, dim=1).get_fdata()
    fill_hole_dim_3 = fill_holes(sample_singlelabel_segmentation, dim=2).get_fdata()

    contiguous_holes, _ = sp.ndimage.label(fill_holes_dim_1)
    contiguous_holes_dim3, _ = sp.ndimage.label(fill_hole_dim_3)

    assert np.sum(contiguous_holes_dim3, axis=(0,1,2)) <= np.sum(contiguous_holes, axis=(0,1,2))



def test_fill_holes_filling_hole(sample_image_with_hole, sample_image_without_hole):
    """
    Giving as input a segmentation with a a 3x3x3 centered cube made of 1s
    with a hole in the center (label 0), performs the fill_holes functionon.

    tests:
    - If the output represents an image with the center hole filled
    """ 

    fill_holes_mask = fill_holes(sample_image_with_hole).get_fdata()

    assert np.all(fill_holes_mask == sample_image_without_hole.get_fdata())


def test_fill_holes_filling_greater_hole(sample_image_with_hole_9, sample_image_without_hole_9):
    """
    Giving as input a segmentation with a a 3x3x3 centered cube made of 1s
    with a hole in the center (label 0), performs the fill_holes functionon

    tests:
    - If the output represents an image with the center hole filled
    """ 

    fill_holes_mask = fill_holes(sample_image_with_hole_9).get_fdata()

    assert np.all(fill_holes_mask == sample_image_without_hole_9.get_fdata())