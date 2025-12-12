import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

def extract_features(image_path):

    # Read image
    img = cv2.imread(image_path)

    # Resize & convert to grayscale
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ----------------------------
    # 1️⃣ HOG FEATURE EXTRACTION
    # ----------------------------
    hog_features, _ = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True
    )

    # ----------------------------
    # 2️⃣ LBP FEATURE EXTRACTION
    # ----------------------------
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    (hist, _) = np.histogram(lbp, bins=59, range=(0, 59))
    hist = hist.astype("float")
    hist /= hist.sum()

    # ----------------------------
    # 3️⃣ STATISTICAL FEATURES
    # ----------------------------
    mean = np.mean(gray)
    std = np.std(gray)
    var = np.var(gray)

    # Combine all features
    feature_vector = np.hstack([hog_features, hist, mean, std, var])

    return feature_vector
