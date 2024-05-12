import matplotlib.pylab as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, average_precision_score
from easydict import EasyDict as edict
from typing import Callable
from tqdm import tqdm


class Metric:
    def evaluate_ood(self, anomaly_score, ood_gts, verbose=True):

        ood_gts = ood_gts.squeeze()
        anomaly_score = anomaly_score.squeeze()

        ood_mask = (ood_gts == 1)
        ind_mask = (ood_gts == 0)

        ood_out = anomaly_score[ood_mask]
        ind_out = anomaly_score[ind_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))

        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        if verbose:
            print(f"Calculating Metrics for {len(val_out)} Points ...")

        auroc, aupr, fpr = self.calculate_ood_metrics(val_out, val_label)

        if verbose:
            print(f'Max Logits: AUROC score: {auroc}')
            print(f'Max Logits: AUPRC score: {aupr}')
            print(f'Max Logits: FPR@TPR95: {fpr}')

        result = {
            'auroc': auroc,
            'aupr': aupr,
            'fpr95': fpr
        }

        return result

    def calculate_ood_metrics(self, out, label):

        # fpr, tpr, _ = roc_curve(label, out)

        prc_auc = average_precision_score(label, out)
        roc_auc, fpr, _ = self.calculate_auroc(out, label)
        # roc_auc = auc(fpr, tpr)
        # fpr = fpr_at_95_tpr(out, label)

        return roc_auc, prc_auc, fpr

    def calculate_auroc(self, conf, gt):
        fpr, tpr, threshold = roc_curve(gt, conf)
        roc_auc = auc(fpr, tpr)
        fpr_best = 0
        # print('Started FPR search.')
        for i, j, k in zip(tpr, fpr, threshold):
            if i > 0.95:
                fpr_best = j
                break
        # print(k)
        return roc_auc, fpr_best, k


def show_grays(images, cols=2):
    plt.rcParams['figure.figsize'] = (15, 20)
    imgs = images['image'] if isinstance(images, dict) else images

    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, ax = plt.subplots(ncols=cols, nrows=np.ceil(len(imgs) / cols).astype(np.int8), squeeze=False)
    for i, img in enumerate(imgs):
        ax[i // cols, i % cols].imshow(np.asarray(img), cmap='gray')
        ax[i // cols, i % cols].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if isinstance(images, dict): ax[i // cols, i % cols].title.set_text(images['name'][i])
    plt.show()


def rescale(image, path):
    plt.imsave(path, image, cmap='gray')


class OODEvaluator:

    def __init__(self, model):

        self.model = model

    def calculate_auroc(self, conf, gt):
        fpr, tpr, threshold = roc_curve(gt, conf)
        roc_auc = auc(fpr, tpr)
        fpr_best = 0
        # print('Started FPR search.')
        for i, j, k in zip(tpr, fpr, threshold):
            if i > 0.95:
                fpr_best = j
                break
        # print(k)
        return roc_auc, fpr_best, k

    def calculate_ood_metrics(self, out, label):

        prc_auc = average_precision_score(label, out)
        roc_auc, fpr, _ = self.calculate_auroc(out, label)
        return roc_auc, prc_auc, fpr

    def evaluate_ood(self, anomaly_score, ood_gts, verbose=True):

        ood_gts = ood_gts.squeeze()

        anomaly_score = anomaly_score.squeeze()

        ood_mask = (ood_gts == 1)
        ind_mask = (ood_gts == 0)

        ood_out = anomaly_score[ood_mask]
        ind_out = anomaly_score[ind_mask]

        ood_label = np.ones(len(ood_out))
        ind_label = np.zeros(len(ind_out))

        val_out = np.concatenate((ind_out, ood_out))
        val_label = np.concatenate((ind_label, ood_label))

        if verbose:
            print(f"Calculating Metrics for {len(val_out)} Points ...")

        auroc, aupr, fpr = self.calculate_ood_metrics(val_out, val_label)

        if verbose:
            print(f'Max Logits: AUROC score: {auroc}')
            print(f'Max Logits: AUPRC score: {aupr}')
            print(f'Max Logits: FPR@TPR95: {fpr}')

        result = {
            'auroc': auroc,
            'aupr': aupr,
            'fpr95': fpr
        }

        return result

    def compute_anomaly_scores(
            self,
            loader,
    ):

        anomaly_score = []
        ood_gts = []

        for image, gt, ignore, file in tqdm(loader, desc="Dataset Iteration"):
            ood_gts.extend([gt.numpy()])

            score = self.model(image, 0.7)  # -> (H, W)

            anomaly_score.extend([score.cpu().numpy()])

        ood_gts = np.array(ood_gts)
        anomaly_score = np.array(anomaly_score)

        return anomaly_score, ood_gts

    def evaluate_ood_bootstrapped(
            self,
            loader=None,
    ):
        results = edict()

        anomaly_score, ood_gts = self.compute_anomaly_scores(
            loader=loader,
        )

        metrics = self.evaluate_ood(
            anomaly_score=anomaly_score,
            ood_gts=ood_gts,
            verbose=False
        )

        for k, v in metrics.items():
            if k not in results:
                results[k] = []
            results[k].extend([v])

        means = edict()
        stds = edict()
        for k, v in results.items():
            values = np.array(v)
            means[k] = values.mean() * 100.0
            stds[k] = values.std() * 100.0

        return means, stds
