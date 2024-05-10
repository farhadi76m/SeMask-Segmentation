import argparse

import cv2
import numpy as np
import os
import sys
import tqdm
import torch

sys.path.insert(1, os.path.join(sys.path[0], '/content/SeMask-Segmentation/SeMask-Mask2Former'))
sys.path.insert(1, os.path.join(sys.path[0], '/content/'))

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import DefaultPredictor

from mask2former import add_maskformer2_config

import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser(description="Set up and run the segmentation model")
    parser.add_argument("--config_path", type=str,
                        default="/content/SeMask-Segmentation/SeMask-Mask2Former/configs/cityscapes/semantic-segmentation/semask_swin/maskformer2_semask_swin_large_IN21k_384_bs16_90k.yaml",
                        help="Path to the configuration file.")
    parser.add_argument("--weights_path", type=str, default="/content/semask_large_mask2former_cityscapes.pth",
                        help="Path to the model weights.")
    parser.add_argument("--image",
                        help="Path of your image")

    return parser.parse_args()


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_path)
    cfg.merge_from_list(['MODEL.WEIGHTS', args.weights_path])
    cfg.freeze()
    return cfg


class Model:
    def __init__(self, args):
        cfg = setup_cfg(args)
        self.model = DefaultPredictor(cfg)

    def get_predictions(self, image):

        segmentation, mask_cls_result, mask_pred_result = self.model(image)
        energy = mask_cls_result.logsumexp(1).cpu().numpy()
        mask_cls_score = F.softmax(mask_cls_result).cpu().numpy()
        masks = mask_pred_result.sigmoid().cpu().numpy()
        return energy, mask_cls_score, masks

    def process_predictions(self, img, T=0.80):
        # Get model predictions
        energy, mask_cls_score, masks = self.get_predictions(img)
        scores = np.zeros_like(masks[0])
        high_energy_indices = np.where(energy > np.quantile(energy, T))[0]

        for idx in high_energy_indices:
            if (mask_cls_score[idx].argmax() != 19):
                scores = np.maximum(scores, masks[idx] * mask_cls_score[idx].max())

        tpp = np.where(mask_cls_score[..., :-1].max(1)[1] == 0)
        idx_tpp = mask_cls_score[tpp[0], :-1].max(1)[0].argsort(descending=True)[0]
        tcar = masks[[tpp[0]]]
        scores = np.minimum(tcar[idx_tpp], scores)
        return scores


if __name__ == "__main__":
    args = get_args()
    cfg = setup_cfg(args)
    model = Model(cfg)
    img = cv2.imread(args.image)  # Adjust as necessary
    processed_output = model.process_and_visualize_predictions(img)
    cv2.imwrite('output.png',processed_output)
    # Optionally visualize or process `processed_output` further
