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
def sample_multilabel_segmentation(tmp_path):
    sample_file_path = os.path.join('data', 'segmentation.nii.gz.gz')
    return nib.load(sample_file_path)


@pytest.fixture
def sample_medical_image(tmp_path):
    sample_file_path = os.path.join('data', 'CT.nii.gz')
    return nib.load(sample_file_path)



def test_crop_invalid_shape(sample_medical_image, sample_multilabel_segmentation):
    """
    Gets as input an image and a multilabel segmentation, and the invalid 
    shape "wrong" is passed.

    Tests:
    If the function gets an invalid shape, and raises an error.
    """

    with pytest.raises(ValueError, match="Shape invalid."):
        crop(sample_medical_image, sample_multilabel_segmentation, 'wrong')