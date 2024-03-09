import nibabel as nib


import numpy as np
import os
import pytest


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 


from lib.allign_crop import alignment_crop



@pytest.fixture
def sample_multilabel_segmentation(tmp_path):
    sample_file_path = os.path.join('data', 'segmentation.nii.gz')
    return nib.load(sample_file_path)


@pytest.fixture
def sample_medical_image(tmp_path):
    sample_file_path = os.path.join('data', 'PT.nii.gz')
    return nib.load(sample_file_path)



def test_align_crop_empty_input():
    """
    Tests:
    If the function gets an empty input
    """

    empty_image = nib.Nifti1Image(np.zeros((10, 10, 10)), np.eye(4))

    with pytest.raises(ValueError, match="Shape invalid."):
        alignment_crop(empty_image, empty_image, 'invalid shape')