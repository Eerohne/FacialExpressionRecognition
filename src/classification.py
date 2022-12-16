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


# img = cv.imread(assetFolder + "anger\\Training_434813.jpg")
# img = normalizeImg(img)
#
# visual, features = compute_feature(img)
# plt.imshow(img, cmap="gray")
# plt.show()
# plt.imshow(visual, cmap="gray")
# plt.show()

# create figure
# fig = plt.figure()
#
#
# # Adds a subplot at the 1st position
# fig.add_subplot(2, 2, 1)
#
# # showing image
# plt.imshow(img, cmap="gray")
# plt.axis('off')
# plt.title("Image")
#
# # Adds a subplot at the 2nd position
# fig.add_subplot(2, 2, 2)
#
# # showing image
# plt.imshow(visual, cmap="gray")
# plt.axis('off')
# plt.title("HOG Features")
#
# plt.show()
#


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


#
# Main function of this class
#
def main():
    fer_train = "../assets/fer/train/"
    fer_test = "../assets/fer/test/"

    # Compute features on train set on whole images
    print("Compute full features")
    # X_full_hog_train, X_full_lbp_train, y_full_train = extract_full_features(dataset_path=fer_train, compute_hog=True,
    #                                                                          hog_features_name="X_full_hog_train",
    #                                                                          compute_lbp=False)
    X_full_hog_train = pickle.load(open("../assets/features/X_full_hog_train.pkl", "rb"))
    y_full_train = pickle.load(open("../assets/features/y_full.pkl", "rb"))

    # Compute features on train set on salient part of images
    print("Compute salient features")
    X_salient_hog_train, X_salient_lbp_train, y_salient_train = extract_salient_features(dataset_path=fer_train,
                                                                                         compute_hog=True,
                                                                                         hog_features_name="X_salient_hog_train",
                                                                                         compute_lbp=False)
    # X_salient_hog_train = pickle.load(open("../assets/features/salient_backup/X_salient_hog_train.pkl", "rb"))
    # y_salient_train = pickle.load(open("../assets/features/salient_backup/y_salient.pkl", "rb"))

    print("Starting SVM training")
    # Train svm on full images (HOG)
    svm_full, full_fit_time = classify(X_full_hog_train, y_full_train)
    print("Training time for full set is: " + str(full_fit_time))
    # Train svm on salient areas (HOG)
    svm_sal, sal_fit_time = classify(X_salient_hog_train, y_salient_train)
    print("Training time for salient set is: " + str(sal_fit_time))

    # Save the model
    pickle.dump(svm_full, open("../assets/svm/svm_hog_full.pkl", "wb"))
    pickle.dump(svm_sal, open("../assets/svm/svm_hog_sal.pkl", "wb"))

    y_full_train_pred = svm_full.predict(X_full_hog_train)
    y_salient_train_pred = svm_sal.predict(X_salient_hog_train)

    accuracy_score_full_train = accuracy_score(y_full_train_pred, y_full_train)
    accuracy_score_sal_train = accuracy_score(y_salient_train_pred, y_salient_train)

    print("Full images accuracy on training set: " + str(accuracy_score_full_train))
    ConfusionMatrixDisplay.from_predictions(y_full_train_pred, y_full_train)
    plt.show()

    print("Salient images accuracy on training set: " + str(accuracy_score_sal_train))
    ConfusionMatrixDisplay.from_predictions(y_salient_train_pred, y_salient_train)
    plt.show()


if __name__ == "__main__":
    main()
