import cv2 as cv
import dlib
import numpy as np
import matplotlib.pyplot as plt
from assetloader import preprocess_img

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../assets/landmarkModel/shape_predictor_68_face_landmarks.dat')

# img = cv.imread("../assets/ck+/ck/CK+48/anger/S010_004_00000017.png", cv.IMREAD_GRAYSCALE)
img = cv.imread("../assets/ck+/ck/CK+48/anger/S042_004_00000020.png", cv.IMREAD_GRAYSCALE)
img = preprocess_img(img)

# Convert the img color to grayscale
#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Detect the face
rects = detector(img, 1)
# Detect landmarks for each face
for rect in rects:
    # Get the landmark points
    shape = predictor(img, rect)
    # Convert it to the NumPy Array
    shape_np = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        shape_np[i] = (shape.part(i).x, shape.part(i).y)
    shape = shape_np

    # Display the landmarks
    for i, (x, y) in enumerate(shape):
    # Draw the circle to mark the keypoint 
        cv.circle(img, (x, y), 1, (0, 0, 255), -1)
    
# cv.imwrite("yep.png", img)
# cv.imshow("landmarks", img)
# cv.waitKey(0)

plt.imshow(img, cmap="gray")
plt.show()

