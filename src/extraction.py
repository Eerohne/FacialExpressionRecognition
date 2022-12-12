from skimage.feature import hog
from scipy.io import savemat
from preprocessing import preprocess
import numpy as np


#
# Take image and returns the features contained within it
#
def compute_feature(img):
    hog_feature = hog(img, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=False, feature_vector=True)
    return hog_feature


#
# This function extracts the features of one emotion only and saves them into one .mat file with the format:
# <emotion>.mat
#
def extract_emotion_features(emotion, salient_areas):
    emotion_dict = {}
    for i in range(0, len(salient_areas)):
        feature_dict = {}
        for area, imgreg in salient_areas[i].items():
            feature = compute_feature(imgreg)
            feature_dict[area] = feature

        feature_matrix = list(feature_dict.values())
        emotion_dict["img" + str(i + 1)] = np.array(feature_matrix)

    return emotion_dict


#
# Extract features from dataset and saves them to storage for access by matlab script
#
# emotions contains the labels to be detected by our model
# sout is the path to which the features will be exported
# debug activates the debugging mode and does not run the actual feature extraction part of the method
# salient_areas is only used in debug mode
#
def extract_features(emotions=["anger", "contempt", "disgust", "fear", "happiness", "sadness", "surprise"],
                     sout="../assets/matfeatures/",
                     debug=False,
                     salient_areas=None,
                     debug_emo="anger"):
    if debug:
        if salient_areas is None:
            salient_areas = preprocess(debug_emo, use_optimization=True)

        anger_features = extract_emotion_features(debug_emo, salient_areas)
        savemat(sout + debug_emo + ".mat", anger_features)

    else:
        for emotion in emotions:
            salient_areas = preprocess(emotion=emotion, use_optimization=True)
            print("Start extracting features for: " + emotion)
            emotion_features = extract_emotion_features(emotion, salient_areas)
            savemat(sout + emotion + ".mat", emotion_features)
            print("Done extracting features for: " + emotion)
