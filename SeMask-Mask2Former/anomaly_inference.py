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

    def get_predictions(self, image, output_cls=False):
        if output_cls:
            segmentation, mask_cls_result, mask_pred_result, output_cls_result = self.model(image)
            return segmentation, mask_cls_result, mask_pred_result, output_cls_result
        else:
            segmentation, mask_cls_result, mask_pred_result = self.model(image)
            return segmentation, mask_cls_result, mask_pred_result

    def process_and_visualize_predictions(self, img, T=8):
        # Get model predictions
        predictions = self.get_predictions(img, False)

        # Initialize a map for tracking positives
        map_positives = torch.ones_like(predictions[2][0])

        # Class probabilities using softmax
        cls = F.softmax(predictions[1], 1)

        # Handling top predictions for specific class (e.g., 13)

        # Anomaly detection
        energy = predictions[1].logsumexp(1).cpu().numpy()
        high_energy = np.where(energy > np.quantile(energy, T))[0]
        if len(high_energy) > 0:
            for idx in high_energy :
                if cls[idx].argmax() != 19 :
                    # att_map = 1 - predictions[2][anomalies.copy()].sigmoid().max(0)[0]
                    map_positives = torch.minimum(map_positives, 1 - predictions[2][idx].sigmoid() * cls[idx].max())



        # Another specific class handling (e.g., 0)
        tpp = torch.where(cls[..., :-1].max(1)[1] == 0)
        idx_tpp = cls[tpp[0], :-1].max(1)[0].argsort(descending=True)[0]
        tcar = predictions[2][[tpp[0]]].sigmoid()
        map_positives = torch.minimum(1 - tcar[idx_tpp], map_positives).cpu().numpy()

        # max_indices = np.argmax(cls.cpu().numpy(), axis=1)
        max_indices = np.argmax(cls.cpu().numpy(), axis=1)
        positive = predictions[2].sigmoid()[max_indices != 19.0].cpu().numpy()

        # # remove the borders of the non-void predictions up to epsilon=0.001
        for i in range(positive.shape[0]):
            for j in range(i + 1, positive.shape[0]):
                neg_border = np.clip((1 - np.logical_and(positive[i] > 0.1, positive[j] > 0.1)) + 0.0, 0, 1)
                map_positives = np.minimum(neg_border, map_positives)

        # Visualize the map of positives
        # plt.imshow(map_positives)
        # plt.show()

        return map_positives


if __name__ == "__main__":
    args = get_args()
    cfg = setup_cfg(args)
    model = Model(cfg)
    img = cv2.imread(args.image)  # Adjust as necessary
    processed_output = model.process_and_visualize_predictions(img)
    cv2.imwrite('output.png', processed_output)
    # Optionally visualize or process `processed_output` further
