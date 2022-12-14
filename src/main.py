# example usage of preprocessing.py
#from preprocessing import preprocess  # import preprocess()
from preprocessing_single import PreProcessor  # import preprocess()
import cv2 as cv
from extraction import extract_features  # import extract_features()


emo = "anger"

# these are the default parameters of preprocess
# def preprocess(detector=dlib.get_frontal_face_detector(), 
#                predictor=dlib.shape_predictor('assets/landmarkModel/shape_predictor_68_face_landmarks.dat'),
#                img_size=128,
#                dataset="assets\\ck+_128",
#                emotion="anger",
#                output_size=64,
#                use_optimization = False,
#                write_dir=None):
# 
# if use_optimization is false, there is a manual 20 pixel radius bounding box that is used
# 
# returns:
# salient_areas will be an array of dictionaries, each one representing an image "img"
#   img = {"leye": <left_eye_salient_area>, "reye": <>, "mouth": <>}
#   salient_areas = [img1, img2, img3,..., imgn]
# salient_areas = preprocess(emotion="surprise", use_optimization=True)

# single example
img = cv.imread(r"C:\Users\Bryan\Documents\_McGill U3\sem2\COMP558\project\EmotionRecognition\assets\ck+_128\anger\S010_004_00000019.png")
preProcessor = PreProcessor()
sareas = preProcessor.preprocess(img) # np.array of black image w/ salient areas cropped





#
# Testing the extraction of features
#
#extract_features(salient_areas=salient_areas, debug=True, debug_emo=emo)

# Extract all features from dataset. Each emotion will have its own .mat file generated that will contain all the features within its corresponding images
extract_features()
