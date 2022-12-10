import numpy as np
import cv2 as cv
import pickle as pk
import matplotlib.pyplot as plt
import os


# Function that blurs and equalizes the image before returning it
def preprocess_img(img):
    img = cv.GaussianBlur(img, (3, 3), 1)
    img = cv.equalizeHist(img)
    return img


# Open Image and convert it to grayscale (the images are already in greyscale, but I am doing this to avoid any issue)
# img2 = cv.imread("../assets/ck+/ck/CK+48/anger/S029_001_00000018.png", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("../assets/ck+/ck/CK+48/anger/S042_004_00000020.png", cv.IMREAD_GRAYSCALE)

print(img2.shape)

# Plot the image
plt.imshow(img2, cmap="gray")
plt.show()

im = cv.GaussianBlur(img2, (3, 3), 1)
plt.imshow(im, cmap="gray")
plt.show()

# histogram equilization
dist = cv.equalizeHist(im)

plt.imshow(dist, cmap="gray")
plt.show()

img3 = preprocess_img(cv.imread("../assets/ck+/ck/CK+48/anger/S042_004_00000020.png", cv.IMREAD_GRAYSCALE))
plt.imshow(img3, cmap="gray")
plt.show()


# List of emotions. Each emotion in the model will be represented by its index in  this list
emotions = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

# Folder containing the dataset
asset_folder = "../assets/ck+/ck/CK+48"

dataset = []
# Load dataset
for emotion in emotions:
    path = os.path.join(asset_folder, emotion)
    img_emotion = emotions.index(emotion)
    for img_name in os.listdir(path):
        img = cv.imread(os.path.join(path, img_name), cv.IMREAD_GRAYSCALE)
        img = preprocess_img(img)
        dataset.append([img_emotion, img])


print(len(dataset))


hog = cv.HogDescriptor()


