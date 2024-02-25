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
def sample_multilabel_segmentation(tmp_path):
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
    sample_array = sample_multilabel_segmentation.get.fdata()

    assert isinstance(result, nib.Nifti1Image)
    assert result_array.shape == sample_array.shape
    assert result.get_data_dtype() == np.uint8
