import cv2 as cv
import dlib
import numpy as np
import matplotlib.pyplot as plt
import os

# globals
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('assets/landmarkModel/shape_predictor_68_face_landmarks.dat')
img_size = 128
dataset = "assets\\ck+_128"
emotion = "anger"
use_optimization = False

(rows, cols) = (img_size, img_size)

# Function that blurs and equalizes the image before returning it
def preprocessImg(img):
    img = cv.GaussianBlur(img, (3, 3), 1)
    img = cv.equalizeHist(img)
    return img

def getLoi(landmarks):
    # landmarks of interest
    # - left eye left corner:36
    # - left eye right corner: 39
    # - right eye left corner: 42
    # - right eye right  corner: 45
    # - top lip bottom: 62
    # - bottom lip top: 66
    # - nose top: 27
    loi = {}
    loi["leye"] = (landmarks[36]+landmarks[39])//2
    loi["reye"] = (landmarks[42]+landmarks[45])//2
    loi["mouth"] = (landmarks[62]+landmarks[66])//2
    loi["lnose"] = (landmarks[39]+landmarks[27])//2
    loi["rnose"] = (landmarks[42]+landmarks[27])//2
    loi["llbrow"] = landmarks[17]
    loi["lrbrow"] = landmarks[19]
    loi["ltbrow"] = landmarks[21]
    loi["rlbrow"] = landmarks[22]
    loi["rtbrow"] = landmarks[24]
    loi["rrbrow"] = landmarks[26]
    return loi

def plotLandmarks(img, loi):
    newimg = img.copy()
    # plot our landmarks of interest
    for vals in loi.values():
        x = vals[0]
        y = vals[1]
        cv.circle(newimg, (x, y), 1, (0, 0, 255), -1)

    return newimg

def getLandmarks(img):
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
    
    return shape

def hashDistance(img1, img2):
    #showTwoImages(img1, img2, "before resize")
    # resize to 8x8
    img1 = cv.resize(img1, (8, 8), interpolation=cv.INTER_AREA)
    img2 = cv.resize(img2, (8, 8), interpolation=cv.INTER_AREA)

    # avg intensity
    m1 = img1.mean()
    m2 = img2.mean()

    #print(m1, m2)
    #showTwoImages(img1, img2, "after resize")

    # threshold image, 255 for white pixel (visualization   )
    img1 = np.where(img1 > m1, 1, 0).astype(np.uint8)
    img2 = np.where(img2 > m2, 1, 0).astype(np.uint8)
    #showTwoImages(img1, img2, "after treshhold")
    
    # hamming distance
    hd = np.count_nonzero(img1!=img2)
    #   print(hd)
    return hd


def transform(img):
    landmarks = getLandmarks(img)
    landmarks =  np.int_(np.c_[ landmarks, np.ones(68) ])

    loi = getLoi(landmarks)


    # loiimg = plotLandmarks(img, loi)
    # plt.imshow(loiimg, cmap="gray")
    # plt.show()

    # face alignment
    theta = np.arctan((loi["reye"][1]-loi["leye"][1])/(loi["reye"][0]-loi["leye"][0])) # rads
    theta = theta*180/np.pi # to degs

    # rotation matrix
    R = cv.getRotationMatrix2D((cols/2,rows/2),theta,1) 

    # rotate to upright face
    img = cv.warpAffine(img,R,(cols,rows)) 

    newlandmarks = getLandmarks(img)
    loi = getLoi(newlandmarks)

    # rotate_img = plotLandmarks(img, loi)
    # plt.imshow(rotate_img, cmap="gray")
    # plt.show()

    # normalization
    targetx = 0.29*img_size
    targety = 0.37*img_size
    scalex = targetx/loi["leye"][0]
    scaley = targety/loi["leye"][1]
    img = cv.resize(img, None, fx=scalex, fy=scaley)
    newlandmarks = getLandmarks(img)
    loi = getLoi(newlandmarks)

    # scale_img = plotLandmarks(img, loi)
    # plt.imshow(scale_img, cmap="gray")
    # plt.show()

    return img, loi

def getOptimizedRegion(imgs):
    
    # calculate bounding distances for active regions, ie. left/right eye, mouth
    regions = ["leye", "reye", "mouth"]
    bounds = {}
    optimal_l = {}
    for region in regions:
        currmin = img_size # max image size
        # iterate through all image pairs (emotion, neutral)
        for img_pair in imgs:
            # iterate through the pair (emotion --> neutral)
            # compute minimum bounding distance
            for (img, loi) in img_pair:
                d = []
                if(region == "reye"):
                    d.append(loi[region][0]-loi["lnose"][0]) # x-distance to edges
                else:
                    d.append(loi[region][0])
                if(region == "leye"):
                    d.append(loi["rnose"][0]-loi[region][0])
                else:
                    d.append(cols-loi[region][0]) 
                d.append(loi[region][1]) # y-distance to edges
                d.append(rows-loi[region][1])
                currmin = min(currmin, min(d))
            
        bounds[region]= currmin
        print(currmin)
        similarities = []
        # iterate through all image pairs (emotion, neutral)
        for img_pair in imgs:
            similarity = []
            # iterate through the pair (emotion --> neutral)
            for l in range(2, bounds[region]+1):
                hash=0
                (img1, loi1) = img_pair[0]
                left1 = loi1[region][0] - l
                right1 = loi1[region][0] + l
                top1 = loi1[region][1] - l
                bot1 = loi1[region][1] + l

                (img2, loi2) = img_pair[1]
                left2 = loi2[region][0] - l
                right2 = loi2[region][0] + l
                top2 = loi2[region][1] - l
                bot2 = loi2[region][1] + l
            
                hash = hashDistance(img1[top1:bot1, left1:right1], img2[top2:bot2, left2:right2])
                similarity.append(1/(hash+1))
            similarities.append(similarity)
        # average of similarities across people
        similarities = np.array(similarities).mean(axis=0)
        print(similarities)
        # choose bounding size that maximizes similarity
        optimal_l[region] = np.argmax(similarity)+2 # indexed by 0

    print(optimal_l)

    return optimal_l

def showImages(imgs, title=""):
    numimg = len(imgs)
    plt.title(title)

    for i, img in enumerate(imgs):
        plt.subplot(1, numimg, i+1)
        plt.imshow(img, cmap="gray")

    plt.show()

# check b/w neutral and angry
def main():
    imgs = []
    processed = []

    print("Starting pre-processing")
    if not os.path.exists(dataset):
        print("No such dataset path")
        return

    # load images
    print("Loading dataset images")
    for subdir, dirs, files in os.walk(dataset+"\\"+emotion):
        for file in files:
            # for every emotion, find neutral emotion
            subject = file.split("_")[0]
            i = 1
            neutral_file = dataset + "\\" + "neutral" + "\\" + f"{subject}_00{i}_00000001.png"
            while(not os.path.exists(neutral_file) or i > 1000): # change to exception subject not found
                i+=1
                neutral_file = dataset + "\\" + "neutral" + "\\" + f"{subject}_00{i}_00000001.png"
            current_file = dataset + "\\" + emotion + "\\" + file
            imgs.append((current_file, neutral_file))
    print("Finished loading images")
    
    # pre-process images
    print("Starting image pre-processing")
    for img_pair in imgs:
        processed_pair = []
        for img in img_pair:
            img = cv.imread(img)
            #showImages([img])

            # grayscale
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            #showImages([img])

            # blur and equalize
            img = preprocessImg(img)
            #showImages([img])
            
            # rotate to aligned image, normalize
            (img, loi) = transform(img)
            processed_pair.append((img, loi))
            #showImages([img])
        processed.append(processed_pair)
    print("Finished pre-processing images")

    # can use manual values for cropping
    if use_optimization:
        # get optimized salient areas
        print("Starting optimized region search")
        optimal_l = getOptimizedRegion(processed)
        print("Finished optimized region search")
    else:
        optimal_l = {
            "leye": 20,
            "reye": 20,
            "mouth": 20
        }
    regions = optimal_l.keys()
    
    # derive salient areas from optimized regions
    print("Computing salient areas")
    salient_areas = []
    for (img_meta, na) in processed:
        cimg = {}
        (img, loi) = img_meta

        for region in regions:
            left = loi[region][0] - optimal_l[region]
            right = loi[region][0] + optimal_l[region]
            top = loi[region][1] - optimal_l[region]
            bot = loi[region][1] + optimal_l[region]
            cimg[region] = (img[top:bot, left:right])

        salient_areas.append(cimg)
        # showImages(cimg.values())
    print("Finished salient areas")
    
    # salient areas will be an array of dictionaries
    # img = {"leye": <left_eye_salient_area>, "reye": <>, "mouth": <>}
    # salient_areas = [img1, img2, img3,..., imgn]


if __name__ == "__main__":
    main()
