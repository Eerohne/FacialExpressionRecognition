from sklearn import svm
import cv2 as cv
import os
import pickle
from extraction import extract_full_features, extract_salient_features
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

assetFolder = "..\\assets\\FER\\test\\"


#
# Takes features and labels and trains an SVM model using them
# The model fitting time is also returned
#
def classify(X, y):
    start_time = time.time()
    svm_fer = svm.LinearSVC(max_iter=10000)
    svm_fer.fit(X, y)

    fit_time = time.time() - start_time

    return svm_fer, fit_time


def train_models(train_path):
    # Extract LBP and HOG features from salient areas
    X_salient_hog, X_salient_lbp, y_salient = extract_salient_features(dataset_path=train_path,
                                                                       compute_hog=True,
                                                                       compute_lbp=True,
                                                                       hog_features_name="X_salient_hog_train",
                                                                       lbp_feature_name="X__salient_lbp_train",
                                                                       y_name="y_salient_train"
                                                                       )

    # Extract LBP and HOG features from the full images
    X_full_hog, X_full_lbp, y_full = extract_full_features(dataset_path=train_path,
                                                           compute_hog=True,
                                                           compute_lbp=True,
                                                           hog_features_name="X_salient_hog_train",
                                                           lbp_feature_name="X_lpb_salient_train",
                                                           y_name="y_salient_train"
                                                           )

    # Train svm on full images (HOG)
    print("Starting Full HOG SVM training")
    svm_full_hog, full_fit_time_hog = classify(X_full_hog, y_full)
    print("Training time for full set with HOG is: " + str(full_fit_time_hog))

    # Train svm on full images (LBP)
    print("Starting Full LBP SVM training")
    svm_full_lbp, full_fit_time_lbp = classify(X_full_lbp, y_full)
    print("Training time for full set with LBP is: " + str(full_fit_time_lbp))

    # Train svm on salient areas (HOG)
    print("Starting Salient HOG SVM training")
    svm_sal_hog, sal_fit_time_hog = classify(X_salient_hog, y_salient)
    print("Training time for salient set with HOG is: " + str(sal_fit_time_hog))

    # Train svm on salient areas (LBP)
    print("Starting Salient LBP SVM training")
    svm_sal_lbp, sal_fit_time_lbp = classify(X_salient_lbp, y_salient)
    print("Training time for salient set with LBP is: " + str(sal_fit_time_lbp))

    # Dump all the trained models
    pickle.dump(svm_full_hog, open("../assets/svm/svm_hog_full.pkl", "wb"))
    pickle.dump(svm_full_lbp, open("../assets/svm/svm_lbp_full.pkl", "wb"))
    pickle.dump(svm_sal_hog, open("../assets/svm/svm_hog_salient.pkl", "wb"))
    pickle.dump(svm_sal_lbp, open("../assets/svm/svm_lbp_salient.pkl", "wb"))

    return svm_full_hog, svm_full_lbp, svm_sal_hog, svm_sal_lbp


def test_models(test_path, svm_full_hog, svm_full_lbp, svm_sal_hog, svm_sal_lbp):
    # Extract LBP and HOG features from salient areas
    X_salient_hog, X_salient_lbp, y_salient = extract_salient_features(dataset_path=test_path,
                                                                       compute_hog=True,
                                                                       compute_lbp=True,
                                                                       hog_features_name="X_salient_hog_train",
                                                                       lbp_feature_name="X__salient_lbp_train",
                                                                       y_name="y_salient_train"
                                                                       )

    # Extract LBP and HOG features from the full images
    X_full_hog, X_full_lbp, y_full = extract_full_features(dataset_path=test_path,
                                                           compute_hog=True,
                                                           compute_lbp=True,
                                                           hog_features_name="X_salient_hog_train",
                                                           lbp_feature_name="X_lpb_salient_train",
                                                           y_name="y_salient_train"
                                                           )

    # Predict full models
    y_full_hog = svm_full_hog.predict(X_full_hog)
    y_full_lbp = svm_full_hog.predict(X_full_lbp)

    # Predict salient models
    y_sal_hog = svm_full_hog.predict(X_salient_hog)
    y_sal_lbp = svm_full_hog.predict(X_salient_lbp)

    # Get accuracies of each model
    accuracy_score_full_hog = accuracy_score(y_full_hog, y_full)
    accuracy_score_full_lbp = accuracy_score(y_full_lbp, y_full)
    accuracy_score_sal_lbp = accuracy_score(y_sal_lbp, y_salient)
    accuracy_score_sal_hog = accuracy_score(y_sal_hog, y_salient)

    # Print accuracies
    print("SVM Full HOG Accuracy: " + str(accuracy_score_full_hog))
    print("SVM Full LBP Accuracy: " + str(accuracy_score_full_lbp))
    print("SVM Salient HOG Accuracy: " + str(accuracy_score_sal_hog))
    print("SVM Salient HOG Accuracy: " + str(accuracy_score_sal_lbp))

    return accuracy_score_full_hog, accuracy_score_full_lbp, accuracy_score_sal_hog, accuracy_score_sal_lbp

# Code to show the confusion matrix
# ConfusionMatrixDisplay.from_predictions(y_salient_test_pred, y_salient_test)
# plt.show()
