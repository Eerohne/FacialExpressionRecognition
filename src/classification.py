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


emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

X=[]
y=[]
count = 0

print("Starting Train feature extraction")
for emotion in emotions:
    count_emotion = 1
    for subdir, dirs, files in os.walk(assetFolder + emotion):
        for file in files:
            img = cv.imread(assetFolder + emotion + "\\" + file)
            img = normalizeImg(img)

            hog_vis, hog_features = compute_feature(img)


            X.append(hog_features)
            y.append(emotion)
            count += 1
            count_emotion += 1
    print("Emotion " + emotion + ", Number of Images Total: " + str(count_emotion))

print("Ended feature extraction")
print("Length of X: " + str(len(X)))
print("Length of y: " + str(len(y)))

pickle.dump(X, open("..\\assets\\X.pkl", 'wb'))
pickle.dump(y, open("..\\assets\\y.pkl", 'wb'))


print("Load the dataset")
X = pickle.load(open("..\\assets\\X.pkl", 'rb'))
y = pickle.load(open("..\\assets\\y.pkl", 'rb'))


import time
startTime = time.time()

print("Start to Train SVM")
svm_clf = svm.LinearSVC(max_iter=10000)
svm_clf.fit(X, y)

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))

print("Dumping the model")
pickle.dump(svm_clf, open("..\\assets\\svm.pkl", 'wb'))

print(accuracy_score(svm_clf.predict(X), y))


print("Starting Test feature extraction")
for emotion in emotions:
    count_emotion = 1
    for subdir, dirs, files in os.walk(assetFolder + emotion):
        for file in files:
            img = cv.imread(assetFolder + emotion + "\\" + file)
            img = normalizeImg(img)

            hog_vis, hog_features = compute_feature(img)


            X.append(hog_features)
            y.append(emotion)
            count += 1
            count_emotion += 1
    print("Emotion " + emotion + ", Number of Images Total: " + str(count_emotion))

print("Ended feature extraction")
print("Length of X: " + str(len(X)))
print("Length of y: " + str(len(y)))


svm_clf = pickle.load(open("..\\assets\\svm.pkl", 'rb'))

print(accuracy_score(svm_clf.predict(X), y))


#
# Takes features and labels and trains an SVM model using them
# The model fitting time is also returned
#
def classify(X, y):
    start_time = time.time()
    svm_fer = svm.LinearSVC()
    svm_fer.fit(X, y)

    fit_time = start_time - time.time()

    return svm_fer, fit_time


#
# Main function of this class
#
def main():
    fer_train = "../assets/fer/train/"
    fer_test = "../assets/fer/test/"

    X_full_hog, X_full_lbp, y_full = extract_full_features()
    X_full_hog, X_full_lbp, y_full = extract_salient_features()




if __name__ == "__main__":
    main()

