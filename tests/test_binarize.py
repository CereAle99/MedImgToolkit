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
    sample_file_path = os.path.join('data', 'segmentation_singlelabel.nii.gz')
    return nib.load(sample_file_path)


@pytest.fixture
def sample_multilabel_segmentation():
    sample_file_path = os.path.join('data', 'segmentation.nii.gz')
    return nib.load(sample_file_path)


@pytest.fixture
def sample_image_multi_012():
    """
    Create a 3x3x3 cube with with labels 0, 1, and 2.
    Put it in a NiftiImage instance
    """
    data = np.zeros((3, 3, 3), dtype=np.uint8)
    data[1,:,:] = 1
    data[2,:,:] = 2

    nifti_image = nib.Nifti1Image(data, affine=np.eye(4))

    return nifti_image


@pytest.fixture
def sample_image_multi_100():
    """
    Create a 3x3x3 cube with with labels 0, 1, and 2.
    Put it in a NiftiImage instance
    """
    data = np.zeros((3, 3, 3), dtype=np.uint8)
    data[0,:,:] = 1

    nifti_image = nib.Nifti1Image(data, affine=np.eye(4))

    return nifti_image


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
    binarized_image = binarize(sample_singlelabel_segmentation, 1)

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


def test_binarize_label_zero(sample_image_multi_012, sample_image_multi_100):
    """
    Binarize function gets as input a multilabel segmentation and a label=0

    Tests:
    If the output segmentation has 0s where befor it had other labels and 
    1s where it had 0s.
    """

    binarized_image = binarize(sample_image_multi_012, 0)

    assert np.all(binarized_image.get_fdata() == sample_image_multi_100.get_fdata())