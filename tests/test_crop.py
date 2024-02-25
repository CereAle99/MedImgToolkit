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
    sample_file_path = os.path.join('data', 'segmentation.nii')
    return nib.load(sample_file_path)


@pytest.fixture
def sample_medical_image(tmp_path):
    sample_file_path = os.path.join('data', 'CT.nii')
    return nib.load(sample_file_path)