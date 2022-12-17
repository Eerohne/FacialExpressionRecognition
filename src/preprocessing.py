import os
from itertools import combinations

import cv2 as cv
import dlib
import numpy as np
import matplotlib.pyplot as plt


# PreProcessor class for FER image preprocessing and salient area extraction
class PreProcessor:
    # Default constructor for the PreProcessor class
    #   landmark_model_path: string path to a trained 68-landmark predictor mmodel
    #   optimal_l: dictionary containing optimal bounding box sizes for salient areas. It should contain:
    #              - "leye": integer size for left eye salient area
    #              - "reye": integer size for right eye salient area
    #              - "mouth": integer size for mouth salient area
    def __init__(self, landmark_model_path='assets/landmarkModel/shape_predictor_68_face_landmarks.dat', optimal_l={}):
        self.predictor = dlib.shape_predictor(landmark_model_path)
        self.detector = dlib.get_frontal_face_detector()
        self.optimal_l = optimal_l

    # Method to load file paths for a given dataset
    #   dataset_dir: string path to the root directory for the image dataset
    #   emotion_dir: string name of the emotion to load, default value is None.
    #                If no emotion is provided, the entire dataset is loaded by
    #                grouping all emotions by subject found under the neutral emotion
    #   RETURN: return a list of images to process, each image is a tuple (file_path, loaded_image)
    def loadDataset(self, dataset_dir, emotion_dir=None):
        imgs = []

        if not os.path.exists(dataset_dir):
            print("No such dataset path")
            return

        # load images
        subjects = set([])
        for subdir, dirs, files in os.walk(dataset_dir + "\\" + (emotion_dir if emotion_dir else "neutral")):
            for file in files:
                # Emotion provided, only add this directory
                if emotion_dir:
                    current_file = dataset_dir + "\\" + emotion_dir + "\\" + file
                    curr_img = cv.imread(current_file)  # load image to memory
                    imgs.append((file, curr_img))
                else:  # add emotion images grouped by subject
                    curr_subject_imgs = []
                    subject = file.split("_")[0]

                    # already seen this subject, next
                    if subject in subjects:
                        continue
                    subjects.add(subject)

                    # add neutral emotion
                    current_file = dataset_dir + "\\" + "neutral" + "\\" + file
                    curr_img = cv.imread(current_file)  # load image to memory
                    curr_subject_imgs.append((file, curr_img))

                    # search other emotion directories for subject
                    for subdir2, dirs2, files2 in os.walk(dataset_dir + "\\"):
                        # ignore neutral directory
                        if subdir2 == dataset_dir + "\\" + "neutral":
                            continue
                        for file2 in files2:
                            curr_subject = file2.split("_")[0]
                            # same subject, add to current grouping
                            if curr_subject == subject:
                                current_file2 = subdir2 + "\\" + file2
                                curr_img2 = cv.imread(current_file2)  # load image to memory
                                curr_subject_imgs.append((file2, curr_img2))
                    # add this subject's image grouping to the list
                    imgs.append(curr_subject_imgs)
        return imgs

    # Method to plot landmarks provided onto the image
    #   img: np.array of image
    #   loi: landmark dictionary with keys labelling landmark positions (x, y) on the image
    #   RETURN: copy of img with landmarks drawn
    def plotLandmarks(self, img, loi):
        newimg = img.copy()
        # plot our landmarks of interest
        for vals in loi.values():
            x = vals[0]
            y = vals[1]
            cv.circle(newimg, (x, y), 1, (0, 0, 255), -1)

        return newimg

    # Method to quickly show a list of images
    #   imgs: array of images
    #   title: default as empty, title of plot
    def showImages(self, imgs, title=""):
        numimg = len(imgs)
        plt.title(title)

        for i, img in enumerate(imgs):
            plt.subplot(1, numimg, i + 1)
            plt.imshow(img, cmap="gray")

        plt.show()

    # Method that blurs and equalizes the image before returning it
    #   img: np.array image
    #   RETURN: image that is blurred and histogram equalized
    def normalizeImg(self, img):
        img = cv.GaussianBlur(img, (3, 3), 1)
        img = cv.equalizeHist(img)
        return img

    # Method that filters from a list of landmarks the landmarks of interest for processing
    #   landmarks: 68-landmark array of x-y positions
    #   RETURN: return dictionary with keys labelling the landmarks of interest (xy position)
    def getLoi(self, landmarks):
        # landmarks of interest
        # - left eye left corner:36
        # - left eye right corner: 39
        # - right eye left corner: 42
        # - right eye right  corner: 45
        # - top lip bottom: 62
        # - bottom lip top: 66
        # - nose top: 27
        loi = {}
        loi["leye"] = (landmarks[36] + landmarks[39]) // 2
        loi["reye"] = (landmarks[42] + landmarks[45]) // 2
        loi["mouth"] = (landmarks[62] + landmarks[66]) // 2
        loi["lnose"] = (landmarks[39] + landmarks[27]) // 2
        loi["rnose"] = (landmarks[42] + landmarks[27]) // 2
        return loi

    # Method that detects face for facial landmarks from an image
    #   img: np.array image to detect landmarks
    #   RETURN: returns the detected landmarks as an array of 68-landmark xy positions (x, y)
    def getLandmarks(self, img):
        # detect any faces for more precise detection
        rects = self.detector(img, 1)

        # if no face detected, assume search for entire image
        if not rects:
            rects = [dlib.rectangle(0, 0, len(img), len(img))]
        # detect landmarks for faces detected
        for rect in rects:
            # try and detect landmarks
            detected_landmarks = self.predictor(img, rect)
            # convert to np array for processing
            landmarks = np.zeros((68, 2), dtype="int")
            for i in range(68):
                landmarks[i] = (detected_landmarks.part(i).x, detected_landmarks.part(i).y)

        return landmarks

    # Method to compute the hash distance between two images. Defined in the report
    #   img1, img2: images to compute hash distance from
    #   RETURN: hash distance between the images
    def hashDistance(self, img1, img2):
        # resize to 8x8
        img1 = cv.resize(img1, (8, 8), interpolation=cv.INTER_AREA)
        img2 = cv.resize(img2, (8, 8), interpolation=cv.INTER_AREA)

        # avg intensity
        m1 = img1.mean()
        m2 = img2.mean()

        # threshold image, according to average intensity
        img1 = np.where(img1 > m1, 1, 0).astype(np.uint8)
        img2 = np.where(img2 > m2, 1, 0).astype(np.uint8)

        # compute hamming distance
        hd = np.count_nonzero(img1 != img2)

        return hd

    # Method to correct rotation and do spatial normalization of image
    #   img: np.array image
    #   RETURN: tuple of img that is processed by aligning vertically and spatially normalizing and final landmarks detected
    def transform(self, img):
        img_size = len(img)

        # detect landmarks for alignment
        landmarks = self.getLandmarks(img)
        landmarks = np.int_(np.c_[landmarks, np.ones(68)])  # convert to integer
        # reduce to landmarks of interest
        loi = self.getLoi(landmarks)

        # face alignment
        theta = np.arctan((loi["reye"][1] - loi["leye"][1]) / (loi["reye"][0] - loi["leye"][0]))  # rads
        theta = theta * 180 / np.pi  # to degs
        # rotation matrix
        R = cv.getRotationMatrix2D((img_size / 2, img_size / 2), theta, 1)
        # rotate to upright face
        img = cv.warpAffine(img, R, (img_size, img_size))
        newlandmarks = self.getLandmarks(img)  # get new landmarks
        loi = self.getLoi(newlandmarks)

        # normalization
        targetx = 0.29 * img_size  # target point after scaling
        targety = 0.37 * img_size
        scalex = targetx / loi["leye"][0]  # required scaling
        scaley = targety / loi["leye"][1]
        img = cv.resize(img, None, fx=scalex, fy=scaley, interpolation=cv.INTER_AREA)  # resize
        ylen, xlen = img.shape
        # crop/pad image to 128x128
        if ylen < img_size:
            img = np.append(img, np.zeros((img_size - ylen, xlen), dtype=np.uint8), axis=0)
        else:
            img = img[:img_size, :]
        if xlen < img_size:
            img = np.append(img, np.zeros((img_size, img_size - xlen), dtype=np.uint8), axis=1)
        else:
            img = img[:, :img_size]

        # detect landmarks from normalized face
        newlandmarks = self.getLandmarks(img)
        loi = self.getLoi(newlandmarks)

        return img, loi

    # Method to set the PreProcessor class' optimal bounding box sizes
    #   leye: integer left eye bounding box size
    #   reye: integer right eye bounding box size
    #   mouth: integer mouth bounding box size
    def setOptimalL(self, leye, reye, mouth):
        self.optimal_l["leye"] = leye
        self.optimal_l["reye"] = reye
        self.optimal_l["mouth"] = mouth

    # Method to compute the optimal bounding box sizes based on the dataset loaded. Details in report.
    #   SETS: optimal_l to computed optimal bounding box sizes of salient areas
    #   dataset_dir: string path of directory to root dataset
    #   img_size: regularized dataset image size, integer
    def getOptimizedRegion(self, dataset_dir, img_size):
        print("computing optimal regions")
        # load dataset
        imgs = self.loadDataset(dataset_dir)

        # process images
        imgs = self.preprocess_all_emotions(imgs)

        # calculate bounding distances for active regions, ie. left/right eye, mouth
        regions = ["leye", "reye", "mouth"]
        bounds = {}
        optimal_l = {}

        # compute optimal bounding box sizes for each salient area (left eye, right eye, mouth)
        for i, region in enumerate(regions):
            currmin = img_size  # max image size

            # compute max bounding sizes, taken by finding the minimum bounding distance in all directions
            for subj_imgs in imgs:
                # iterate through all processed emotion images grouped by subject  (neutral, emotion1, emotion2, ...)
                for (file_name, img, loi) in subj_imgs:
                    # determine distances for current image
                    d = []
                    if (region == "reye"):  # right eye is bounded by the left nose contour
                        d.append(loi[region][0] - loi["lnose"][0])  # x-distance to edges
                    else:
                        d.append(loi[region][0])
                    if (region == "leye"):  # left eye is bounded by the right nose contour
                        d.append(loi["rnose"][0] - loi[region][0])
                    else:
                        d.append(img_size - loi[region][0])

                    d.append(loi[region][1])  # y-distance to edges
                    d.append(img_size - loi[region][1])

                    # take minimum distance
                    currmin = min(currmin, min(d))

            # set bounding size for current salient area
            bounds[region] = currmin

            # compute the average similarities to pick optimal l's
            similarities = []  # all similarities across subjects and bounding box sizes
            startingl = 5  # skip initial sizes
            # iterate through all processed emotion images grouped by subject  (neutral, emotion1, emotion2, ...)
            for subj_imgs in imgs:
                similarity = []  # similarity list for all bounding box sizes for the current subject

                # iterate through bounding box sizes up to the computed max bounding box size
                for l in range(startingl, bounds[region] + 1):
                    hash = 0
                    # find all pairwise combinations for subject images
                    pairs = list(combinations(subj_imgs, 2))
                    for (img1, img2) in pairs:
                        (file_name1, img1, loi1) = img1

                        # compute the current bounding box
                        left1 = loi1[region][0] - l
                        right1 = loi1[region][0] + l
                        top1 = loi1[region][1] - l
                        bot1 = loi1[region][1] + l

                        (file_name2, img2, loi2) = img2

                        # compute hash distance for current pair
                        # hash for this subject is calculated as the sum of all hash distances across emotions
                        hash += self.hashDistance(img1[top1:bot1, left1:right1], img2[top1:bot1, left1:right1])
                    similarity.append(1 / (hash + 1))  # compute similarity from hash distance
                similarities.append(similarity)

            # take average simialrity across subjects for every bounding box size l
            avgsimilarities = np.array(similarities).mean(axis=0)
            # choose bounding size that maximizes similarity
            optimal_l[region] = np.argmax(avgsimilarities) + startingl  # indexed by 0

        # set optimal sizes
        self.optimal_l = optimal_l

    # Method that will perform preprocessing on the image dataset and detect landmarks
    #   imgs: list of images tro be preprocessed
    #   RETURN: list of images that have been pre processed, transformed
    def preprocess(self, imgs):
        processed = []

        # pre-process images
        print("Starting image pre-processing")
        for (file_path, img) in imgs:
            # grayscale
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # blur and equalize
            img = self.normalizeImg(img)

            # rotate to aligned image, normalize
            (img, loi) = self.transform(img)

            processed.append((file_path, img, loi))
        print("Finished pre-processing images")

        return processed

    # Method that will perform preprocessing on the image dataset and detect landmarks
    # modified method that processes a different structure
    #   imgs: list of images tro be preprocessed
    #   RETURN: list of images that have been pre processed, transformed
    def preprocess_all_emotions(self, imgs):
        processed = []
        # pre-process images
        print("Starting image pre-processing")
        for sub_imgs in imgs:
            curr_subj = []
            for (file_path, img) in sub_imgs:
                # grayscale
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                # blur and equalize
                img = self.normalizeImg(img)

                # rotate to aligned image, normalize
                (img, loi) = self.transform(img)

                curr_subj.append((file_path, img, loi))
            processed.append(curr_subj)
        print("Finished pre-processing images")

        return processed

    # Method that will perform salient area extraction from a dataset
    #   PRE: assume that optimal bounding box sizes have been computed
    #   dataset_dir: string path of directory to root dataset
    #   emotion_dir: string emotion to be processed
    #   img_size: regularized dataset image size, integer
    #   output_size: regularized image size for outputted extracted salient areas
    #   write_dir: default to None, if specified, salient areas will be written to the specified directory
    #   RETURN: returns list of salient areas as a dictionary labelling salient areas of every image processed
    def extractSalientAreas(self, dataset_dir, emotion_dir, img_size, output_size=64, write_dir=None):
        if write_dir is not None and not os.path.exists(write_dir):
            print("No such write directory path")
            return

        # load images
        imgs = self.loadDataset(dataset_dir, emotion_dir=emotion_dir)

        # process images
        imgs = self.preprocess(imgs)

        # extract salient areas from optimized regions
        print("Computing salient areas")
        regions = self.optimal_l.keys()
        salient_areas = []
        for i, (file_name, img, loi) in enumerate(imgs):
            cimg = {}
            # cimg = np.zeros((img_size, img_size))

            # loop through salient area classes
            for region in regions:
                # calculate bounding box based on optimal size
                left = loi[region][0] - self.optimal_l[region]
                right = loi[region][0] + self.optimal_l[region]
                top = loi[region][1] - self.optimal_l[region]
                bot = loi[region][1] + self.optimal_l[region]
                # should be weithin bounds
                if left < 0 or top < 0 or right >= img_size or bot >= img_size:
                    break

                # crop the image to the salient area
                crop_area = (img[top:bot, left:right])
                # regulairzed to output size
                active_area = cv.resize(crop_area, (output_size, output_size), interpolation=cv.INTER_AREA)
                cimg[region] = active_area  # save salient area for this image
                if write_dir is not None:  # write to file as well
                    (filename, ext) = file_name.split(".")

                    cv.imwrite(f"{write_dir}/{filename}_{region}.{ext}", active_area)
            else:  # if loop didnt break
                salient_areas.append(cimg)

        print("Returning salient areas")

        return salient_areas


def main():
    dataset_dir = "assets\\ck+_128"
    emotion_dir = "fear"
    img_size = 128
    preProcessor = PreProcessor()

    # example using optimized bounding box searching
    preProcessor.getOptimizedRegion(dataset_dir, img_size)
    # example using a preset optimized bounding box sizes
    # preProcessor.setOptimalL(leye=20, reye=20, mouth=20)

    sareas = preProcessor.extractSalientAreas(
        dataset_dir=dataset_dir,
        emotion_dir=emotion_dir,
        img_size=img_size,
        output_size=64,
        write_dir="testdir")


if __name__ == "__main__":
    main()
