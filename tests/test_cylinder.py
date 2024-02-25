import nibabel as nib


import numpy as np
import os
import pytest


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 


from lib.cylinder import cylinder


@pytest.fixture
def sample_singlelabel_segmentation():
    sample_file_path = os.path.join('data', 'segmentation_singlelabel.nii')
    return nib.load(sample_file_path)


def test_cylinder_returns_nifti1image(sample_multilabel_segmentation):
    """
    Tests:
    If the output is a NiftiImage instance
    If the input image has the same resolution of the output image
    If the datatype of the output image is binary
    """
    result = cylinder(sample_multilabel_segmentation)
    result_array = result.get_fdata()
    sample_array = sample_multilabel_segmentation.get_fdata()

    assert isinstance(result, nib.Nifti1Image)
    assert result_array.shape == sample_array.shape
    assert result.get_data_dtype() == np.uint8


def test_cylinder_empty_input():
    """
    Tests:
    If the function gets an empty input
    """

    empty_image = nib.Nifti1Image(np.zeros((100, 100, 100)), np.eye(4))

    with pytest.raises(ValueError, match="Input file is empty"):
        cylinder(empty_image)


def test_cylinder_encloses_mask(sample_singlelabel_segmentation):
    """
    Tests:
    If the output encloses the input image
    """

    cylinder_mask = cylinder(sample_singlelabel_segmentation).get_fdata()
    input_mask = sample_singlelabel_segmentation.get_fdata()

    assert np.all(np.logical_or(np.equal(cylinder_mask, input_mask), cylinder_mask))



def test_cylinder_dilation_bigger(sample_singlelabel_segmentation):
    """
    Tests:
    If the output with dilation is bigger than the input without dilation
    """

    cylinder_mask = cylinder(sample_singlelabel_segmentation).get_fdata()
    cylinder_dilated_mask = cylinder(sample_singlelabel_segmentation, 1).get_fdata()

    count_ones_array1 = np.sum(cylinder_mask == 1)
    count_ones_array2 = np.sum(cylinder_dilated_mask == 1)

    assert count_ones_array1 < count_ones_array2


