import pickle

import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from scipy.io import savemat
import cv2 as cv
import os
from preprocessing import PreProcessor
import numpy as np


#
# Take image and returns the hog features contained within it
# The histogram has 9 bins, and is computed by dividing the image in cells of 8x8 pixels, and then sliding a block of
# size 2x2 cells through the image
#
def compute_hog_feature(img,
                        orientations=9,
                        cell_r=8,
                        block_r=2):
    hog_feature = hog(img, orientations=orientations, pixels_per_cell=(cell_r, cell_r),
                      cells_per_block=(block_r, block_r), visualize=False, feature_vector=True)
    return hog_feature


def compute_lbp_feature(img):
    lbp_img = local_binary_pattern(img, 8, 1)

    lbp_hist, _ = np.histogram(lbp_img.ravel(), bins=np.arange(0, 11), range=(0, 10))

    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-10)
    return lbp_img, lbp_hist


#
# This function extracts the features of one emotion only and saves them into one .mat file with the format:
# <emotion>.mat
#
# def extract_emotion_features(emotion, salient_areas):
#     emotion_dict = {}
#     for i in range(0, len(salient_areas)):
#         feature_dict = {}
#         for area, imgreg in salient_areas[i].items():
#             feature = compute_feature(imgreg)
#             feature_dict[area] = feature
#
#         feature_matrix = list(feature_dict.values())
#         emotion_dict["img" + str(i + 1)] = np.array(feature_matrix)
#
#     return emotion_dict


#
# Extract features from dataset and saves them to storage for access by matlab script
#
# emotions contains the labels to be detected by our model
# sout is the path to which the features will be exported
# debug activates the debugging mode and does not run the actual feature extraction part of the method
# salient_areas is only used in debug mode
#
# def extract_features(emotions=["anger", "contempt", "disgust", "fear", "happiness", "sadness", "surprise"],
#                      sout="../assets/matfeatures/",
#                      debug=False,
#                      salient_areas=None,
#                      debug_emo="anger"):
#     if debug:
#         if salient_areas is None:
#             salient_areas = preprocess(debug_emo, use_optimization=True)
#
#         anger_features = extract_emotion_features(debug_emo, salient_areas)
#         savemat(sout + debug_emo + ".mat", anger_features)
#
#     else:
#         for emotion in emotions:
#             salient_areas = preprocess(emotion=emotion, use_optimization=True)
#             print("Start extracting features for: " + emotion)
#             emotion_features = extract_emotion_features(emotion, salient_areas)
#             savemat(sout + emotion + ".mat", emotion_features)
#             print("Done extracting features for: " + emotion)


#
# Returns the set of features computed on the salient areas of every image
#
def extract_salient_features(emotions=["anger", "neutral", "disgust", "fear", "happiness", "sadness", "surprise"],
                             dataset_path="../assets/FER/train/",
                             features_save_path="../assets/features/",
                             compute_hog=True,
                             hog_features_name="X_hog",
                             compute_lbp=False,
                             lbp_feature_name="X_lbp"):
    # Initialize feature arrays
    X_hog = []  # Contains HOG features for each image
    X_lbp = []  # Contains LBP features for each image
    y = []  # Contains emotion labels for each image

    pre_processor = PreProcessor()
    for emotion in emotions:
        salient_areas = pre_processor.preprocess(
                                        img_size=48,
                                        output_size=10,
                                        dataset=dataset_path,
                                        emotion="anger",
                                        eye_r=8,
                                        mouth_r=12,
                                        use_optimization=False)

        for img_regs in salient_areas:
            # image_features_hog = []
            # image_features_lbp = []

            #for sal_name, sal_img in img_regs.items():
            if compute_hog:
                # image_features_hog = image_features_hog + compute_hog_feature(img=sal_img, cell_r=2).tolist()
                X_hog.append(compute_hog_feature(img=img_regs).tolist())
            if compute_lbp:
                # image_features_lbp = image_features_lbp + compute_lbp_feature(sal_img)
                X_lbp.append(compute_lbp_feature(img_regs))

            # X_hog.append(np.array(image_features_hog))
            # X_lbp.append(np.array(image_features_lbp))
            y.append(emotion)

    # Pickle the computed features to avoid repetitive computations
    if compute_hog:
        pickle.dump(X_hog, open(features_save_path + hog_features_name + ".pkl", "wb"))
    if compute_lbp:
        pickle.dump(X_lbp, open(features_save_path + lbp_feature_name + ".pkl", "wb"))

    pickle.dump(y, open(features_save_path + "y_salient.pkl", "wb"))
    return X_hog, X_lbp, y


#
# Returns the set of features computed on every image in its entirety
#
def extract_full_features(emotions=["anger", "neutral", "disgust", "fear", "happiness", "sadness", "surprise"],
                          dataset_path="../assets/FER/train/",
                          features_save_path="../assets/features/",
                          compute_hog=True,
                          hog_features_name="X_hog",
                          compute_lbp=False,
                          lbp_feature_name="X_lbp"):
    # Initialize feature arrays
    X_hog = []  # Contains HOG features for each image
    X_lbp = []  # Contains LBP features for each image
    y = []  # Contains emotion labels for each image

    pre_processor = PreProcessor()
    for emotion in emotions:
        for subdir, dirs, files in os.walk(dataset_path + emotion):
            for file in files:
                # Read the image and normalize it
                img = cv.imread(dataset_path + emotion + "/" + file)
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                img = pre_processor.normalizeImg(img)

                # Extract features
                y.append(emotion)                           # Save label
                if compute_lbp:
                    X_lbp.append(compute_lbp_feature(img))  # Extract LBP features
                if compute_hog:
                    X_hog.append(compute_hog_feature(img))  # Extract HOG features

    # Pickle the computed features to avoid repetitive computations
    if compute_hog:
        pickle.dump(X_hog, open(features_save_path + hog_features_name + ".pkl", "wb"))
    if compute_lbp:
        pickle.dump(X_lbp, open(features_save_path + lbp_feature_name + ".pkl", "wb"))

    pickle.dump(y, open(features_save_path + "y_full.pkl", "wb"))
    return X_hog, X_lbp, y



if __name__ == "__main__":
    proc = PreProcessor()
    im1 = cv.imread("../assets/FER/anger/")
    im1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    im1 = proc.normalizeImg(im1)
    plt.imshow(im1, cmap="gray")
    plt.show()


    # im1_lbp_im, im1_lbp_feature = compute_lbp_feature(im1)
    #
    # plt.imshow(im1_lbp_im, cmap="gray")
    # plt.show()





