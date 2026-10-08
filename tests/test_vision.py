import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import cv2
from vision import mask_external
BASE=Path(__file__).resolve().parents[1]
class Masks(unittest.TestCase):
    def test_training_and_runtime_regions(self):
        rgb=np.full((480,640,3),200,np.uint8)
        for kind in ['right_edge','cup_background','fixed_roi']:
            output=mask_external(rgb,kind,BASE/'assets/prism_fixed_roi.png')
            if kind=='right_edge':
                keep=np.ones((480,640),bool);keep[:,500:]=False
            elif kind=='cup_background':
                keep=np.ones((480,640),bool);keep[:230,440:]=False;keep[:,500:]=False
            else:keep=cv2.imread(str(BASE/'assets/prism_fixed_roi.png'),0)>0
            self.assertTrue((output[keep]==200).all())
            self.assertTrue((output[~keep]==96).all())
            self.assertTrue((rgb==200).all())
    def test_reject_wrong_dimensions(self):
        with self.assertRaises(ValueError):mask_external(np.zeros((240,320,3),np.uint8),'right_edge')
if __name__=='__main__':unittest.main()
