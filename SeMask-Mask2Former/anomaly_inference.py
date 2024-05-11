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
        mask_cls_score = F.softmax(mask_cls_result, 1).cpu().numpy()
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

        tpp = np.where(mask_cls_score[..., :-1].argmax(1) == 0)[0]
        idx_tpp = mask_cls_score[tpp, :-1].max(1).argsort()[::-1][0]
        road = masks[tpp][idx_tpp]

        scores = np.maximum(road, scores)
        if True :
            semantic = self.semantic_inference(torch.tensor(mask_cls_score), torch.tensor(masks))
            scores = self.refinement(scores,semantic).numpy()
        return scores

    def semantic_inference(self, mask_cls, mask_pred):

        mask_cls_f = mask_cls[..., :-1]
        mask_pred_f = mask_pred
        semseg = torch.einsum("qc,qhw->chw", mask_cls_f, mask_pred_f)
        scores, labels = mask_cls.max(-1)
        mask_pred = mask_pred
        keep = labels.ne(19) & (scores > 0.95) & (labels < 11) & (labels >= 0)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
        semseg = torch.cat((semseg, cur_prob_masks), 0)
        return semseg

    def refinement(self, p, semantic):
        outputs_na = torch.tensor(1 - p)
        if semantic[19:, :, :].shape[0] > 1:
            outputs_na_mask = torch.max(semantic[19:, :, :].unsqueeze(0), axis=1)[0]
            outputs_na_mask[outputs_na_mask < 0.5] = 0
            outputs_na_mask[outputs_na_mask >= 0.5] = 1
            outputs_na_mask = 1 - outputs_na_mask
            outputs_na_save = outputs_na.clone().detach().cpu().numpy().squeeze().squeeze()
            outputs_na = outputs_na * outputs_na_mask.detach()
            outputs_na_mask = outputs_na_mask.detach().cpu().numpy().squeeze().squeeze()
        outputs_na = outputs_na.detach().cpu().numpy().squeeze().squeeze()


if __name__ == "__main__":
    args = get_args()
    cfg = setup_cfg(args)
    model = Model(cfg)
    img = cv2.imread(args.image)  # Adjust as necessary
    processed_output = model.process_and_visualize_predictions(img)
    cv2.imwrite('output.png', processed_output)
    # Optionally visualize or process `processed_output` further
