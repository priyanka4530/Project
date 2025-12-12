import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog, local_binary_pattern
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# ------------------------------
# FEATURE EXTRACTION FUNCTION
# ------------------------------
def extract_features(image_path):

    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1 - HOG Features
    hog_features, _ = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True
    )

    # 2 - LBP Features
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    (hist, _) = np.histogram(lbp, bins=59, range=(0, 59))
    hist = hist.astype("float")
    hist /= hist.sum()

    # 3 - Statistical Features
    mean = np.mean(gray)
    std = np.std(gray)
    var = np.var(gray)

    # Feature Vector (Final)
    feature_vector = np.hstack([hog_features, hist, mean, std, var])

    return feature_vector


# ------------------------------
# LOAD DATASET
# ------------------------------
def load_dataset(dataset_path):

    X = []  # features
    y = []  # labels

    real_path = os.path.join(dataset_path, "real")
    fake_path = os.path.join(dataset_path, "fake")

    print("\nExtracting REAL images...")
    for img_name in tqdm(os.listdir(real_path)):
        img_path = os.path.join(real_path, img_name)
        features = extract_features(img_path)
        X.append(features)
        y.append(0)  # 0 = real image

    print("\nExtracting AI images...")
    for img_name in tqdm(os.listdir(fake_path)):
        img_path = os.path.join(fake_path, img_name)
        features = extract_features(img_path)
        X.append(features)
        y.append(1)  # 1 = AI-generated image

    return np.array(X), np.array(y)


# ------------------------------
# MAIN TRAINING FUNCTION
# ------------------------------
if __name__ == "__main__":

    DATASET_PATH = "dataset"   # folder with real/ and fake/
    MODEL_SAVE_PATH = "model.pkl"

    print("\nLoading dataset...")
    X, y = load_dataset(DATASET_PATH)

    print("\nSplitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nTraining SVM model...")
    svm = SVC(kernel="linear", probability=True)
    svm.fit(X_train, y_train)

    print("\nEvaluating model...")
    predictions = svm.predict(X_test)
    print("\nCLASSIFICATION REPORT:\n")
    print(classification_report(y_test, predictions))

    print("\nSaving model as model.pkl...")
    joblib.dump(svm, MODEL_SAVE_PATH)

    print("\n🎉 Training complete! model.pkl created successfully!")
