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
    sample_file_path = os.path.join('data', 'segmentation_singlelabel.nii')
    return nib.load(sample_file_path)


def test_dilate_returns_nifti1image(sample_singlelabel_segmentation):
    """
    Tests:
    If the output is a NiftiImage instance
    If the input image has the same resolution of the output image
    If the datatype of the output image is binary
    """
    result = dilate(sample_singlelabel_segmentation)
    result_array = result.get_fdata()
    sample_array = sample_singlelabel_segmentation.get_fdata()

    assert isinstance(result, nib.Nifti1Image)
    assert result_array.shape == sample_array.shape
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
    If the output for a higher interation parameter encloses the one with a lower one
    If giving 0 as structuring element raises an error
    """

    dilate_dim_1 = dilate(sample_singlelabel_segmentation, iterations=1).get_fdata()
    dilate_dim_2 = dilate(sample_singlelabel_segmentation, iterations=2).get_fdata()

    with pytest.raises(ValueError, match="Dim 0 for the structuring element"):
        dilate(sample_singlelabel_segmentation, iterations=0)
    assert np.all(np.logical_or(np.equal(dilate_dim_2, dilate_dim_1), dilate_dim_2))


def test_dilate_segmentation_is_bigger():
    """
    Tests:
    If the lebels are more in the output image than the input
    """

    dilate_mask = dilate(sample_singlelabel_segmentation).get_fdata()
    input_mask = sample_singlelabel_segmentation.get_fdata()

    assert np.sum(dilate_mask, axis=(0,1)) > np.sum(input_mask, axis=(0,1))