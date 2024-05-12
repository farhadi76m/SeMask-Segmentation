from sklearn.metrics import roc_curve, auc, average_precision_score
import numpy as np
class Metric :
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
