import argparse

import cv2
import numpy as np
import os
import sys
import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '/content/SeMask-Segmentation/SeMask-Mask2Former'))

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import DefaultPredictor

from mask2former import add_maskformer2_config


def setup_cfg(args=None):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file('/content/SeMask-Segmentation/SeMask-Mask2Former/configs/cityscapes/semantic-segmentation/semask_swin/maskformer2_semask_swin_large_IN21k_384_bs16_90k.yaml')
    cfg.merge_from_list(['MODEL.WEIGHTS',  'semask_large_mask2former_cityscapes.pth'])
    cfg.freeze()
        # return cfg
    return cfg

class Model:
    def __init__(self, args):
        cfg = setup_cfg()
        self.model = DefaultPredictor(cfg)

    def get_predictions(self, image, output_cls = False):
        if output_cls :
            segmentation, mask_cls_result, mask_pred_result, output_cls_result = self.model(image)
        else:
            segmentation, mask_cls_result, mask_pred_result = self.model(image)
