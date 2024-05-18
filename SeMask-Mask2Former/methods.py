import numpy as np
import scipy
import torch
from PIL import Image
from scipy.spatial.distance import mahalanobis
from sklearn.ensemble import IsolationForest
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from scipy.stats import zscore
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor



# load Mask2Former fine-tuned on Cityscapes semantic segmentation
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-cityscapes-semantic")
image = Image.open("/content/4.png")


def inference(image):
    global semantic_map
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Model predicts class_queries_logits of shape `(batch_size, num_queries)`
    # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`

    class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
    masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

    # Scale back to preprocessed image size - (384, 384) for all models
    masks_queries_logits = torch.nn.functional.interpolate(
        masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
    )

    # Remove the null class `[..., :-1]`
    masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
    masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

    # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
    segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
    batch_size = class_queries_logits.shape[0]

    semantic_segmentation = []
    for idx in range(batch_size):
        resized_logits = torch.nn.functional.interpolate(
            segmentation[idx].unsqueeze(dim=0), size=[720, 1280], mode="bilinear", align_corners=False
        )
        semantic_map = resized_logits[0].argmax(dim=0)
        semantic_segmentation.append(semantic_map)

    masks_queries_logits = torch.nn.functional.interpolate(
        masks_queries_logits, size=(720, 1280), mode="bilinear", align_corners=False
    )
    return semantic_map, class_queries_logits, masks_queries_logits


def scorer(selected_masks, class_queries_logits, masks_queries_logits):
    score = 1 - masks_queries_logits[0][selected_masks].sigmoid().max(0)[0]
    masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]

    road = np.where(masks_classes[0].argmax(1) == 0)
    road = 1 - masks_queries_logits[0][road].sigmoid().max(0)[0]
    soft_score = np.minimum(score, road)
    return soft_score


def get_isolation_forest(class_queries_logits):
    # Assuming your matrix is a numpy array named 'data'
    data = class_queries_logits.squeeze()[..., :-1]  # Example data

    # Create IsolationForest model
    clf = IsolationForest(contamination=0.1)  # contamination is the proportion of outliers in the data set
    clf.fit(data)

    # Predict anomalies
    predictions = clf.predict(data)

    # Find indices of anomalies
    anomalies = np.where(predictions == -1)[0]

    return anomalies


def get_mahalanobis(class_queries_logits, t=0.95):
    # Assuming your matrix is a numpy array named 'data'
    # data = np.random.random((100, 20))  # Example data
    data = class_queries_logits.squeeze()[..., :-1]  # Example data
    # Calculate the mean and covariance matrix
    mean = np.mean(data.numpy(), axis=0)
    cov = np.cov(data, rowvar=False)
    inv_covmat = scipy.linalg.inv(cov)

    # Calculate Mahalanobis distance for each row
    mahalanobis_distances = np.array([mahalanobis(row, mean, inv_covmat) for row in data])

    # Set a threshold for Mahalanobis distance
    threshold = np.quantile(mahalanobis_distances, t)
    anomalies = np.where(mahalanobis_distances > threshold)[0]

    return anomalies


def get_zscore(class_queries_logits, t=0.95):
    # Assuming your matrix is a numpy array named 'data'
    # data = np.random.random((100, 20))  # Example data
    data = class_queries_logits.squeeze()[..., :-1]  # Example data
    # Calculate Z-scores
    z_scores = np.abs(zscore(data[..., :-1].numpy(), axis=1))

    # Set a threshold for Z-scores to detect anomalies, e.g., 3 standard deviations
    threshold = 3
    anomalies = np.where(np.any(z_scores > threshold, axis=1))[0]

    return anomalies


def get_knn(class_queries_logits, k=5, t=0.95):
    # Fit k-NN
    # Number of neighbors
    data = class_queries_logits.squeeze()[..., :-1]
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(data)

    # Calculate the distance to the k-th nearest neighbor for each point
    distances, _ = nbrs.kneighbors(data)
    kth_distances = distances[:, k - 1]

    # Set a threshold for anomaly detection
    threshold = np.quantile(kth_distances, t)
    anomalies = np.where(kth_distances > threshold)[0]

    return anomalies,kth_distances


def get_ellips(class_queries_logits):
    data = class_queries_logits.squeeze()[..., :-1]
    cov = EllipticEnvelope(contamination=0.1)
    cov.fit(data)

    # Predict anomalies
    predictions = cov.predict(data)

    # Find indices of anomalies
    anomalies = np.where(predictions == -1)[0]

    return anomalies

def get_lof(class_queries_logits):
    # Local Outlier Factor
    data = class_queries_logits.squeeze()[..., :-1]
    lof = LocalOutlierFactor(contamination=0.05)
    lof_preds = lof.fit_predict(data)
    anomalies = np.where((lof_preds == -1))[0]
    return anomalies