import nibabel as nib


import numpy as np
import os
import pytest


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 


from lib.binarize import binarize



@pytest.fixture
def sample_singlelabel_segmentation():
    sample_file_path = os.path.join('data', 'segmentation_singlelabel.nii')
    return nib.load(sample_file_path)


@pytest.fixture
def sample_multilabel_segmentation():
    sample_file_path = os.path.join('data', 'segmentation.nii')
    return nib.load(sample_file_path)


def test_binarize_returns_nifti1image(sample_multilabel_segmentation):
    """
    Tests:
    If the output is a NiftiImage instance
    If the input image has the same resolution of the output image
    If the datatype of the output image is binary
    """
    result = binarize(sample_multilabel_segmentation)
    result_array = result.get_fdata()
    sample_array = sample_multilabel_segmentation.get_fdata()

    assert isinstance(result, nib.Nifti1Image)
    assert result_array.shape == sample_array.shape
    assert result.get_data_dtype() == np.uint8


def test_binarize_single_label(sample_singlelabel_segmentation):
    """
    Tests:
    If the output of a single-label mask is binary
    """
    binarized_image = binarize(sample_singlelabel_segmentation, 15)

    assert binarized_image.get_data_dtype() == np.uint8


    
def test_binarize_empty_input():
    """
    Tests:
    If the function gets an empty input
    """

    empty_image = nib.Nifti1Image(np.zeros((100, 100, 100)), np.eye(4))

    with pytest.raises(ValueError, match="Input file is empty"):
        binarize(empty_image, label=1)


def test_binarize_no_labels_match(sample_singlelabel_segmentation):
    """
    Tests:
    If the binary mask is empty when giving a non-present label
    """
    non_existing_label = 90
    binarized_image = binarize(sample_singlelabel_segmentation, non_existing_label)
    binary_mask = binarized_image.get_fdata()

    # Assert that 
    assert np.all(binary_mask == 0)