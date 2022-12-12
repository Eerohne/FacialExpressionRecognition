from skimage.feature import hog
from scipy.io import savemat
from preprocessing import main
import numpy as np


# Take image and returns the features contained within it
def computeFeature(img):
    hog_feature = hog(img, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=False, feature_vector=True)
    return hog_feature


# Emotion corresponds to the label of the images.
# Images correspond to a dictionary of salient features of the image
# Saves the image features in a .mat file
def img2mat(emotion, images, matname):
    feature_dict = {}
    for area, imgreg in images.items():
        feature = computeFeature(imgreg)
        feature_dict[area] = feature

    file = "../assets/matfeatures/" + emotion + "/" + matname + ".mat"
    savemat(file, feature_dict)


def extractFeatures(emotion, salient_areas):
    for i in range(0,len(salient_areas)):
        img2mat(emotion, salient_areas[i], str(i))


emotions = ["anger", "contempt", "disgust", "fear", "happiness", "sadness", "surprise"]
for emotion in emotions:
    salient_areas = main(emotion)
    print("Start extractiong features for: " + emotion)
    extractFeatures(emotion, salient_areas)
    print("Done extractiong features for: " + emotion)

print("End of program")

