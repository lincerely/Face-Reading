from __future__ import print_function
import os, sys
import cv2
import dlib
import imutils
from imutils.video import VideoStream
from imutils import face_utils
from imutils.face_utils import FaceAligner
from glob import glob
import numpy as np

#config
USE_REGION = True # use part of the feature to train the svm, e.g. only use mouth feature points
LANDMARK_PATH = "data/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(LANDMARK_PATH)
fa = FaceAligner(predictor, desiredFaceWidth=400)

faceRegions = {
    "eye_left": list(range(36,41+1)),
    "eye_right": list(range(42, 47+1)),
    "nose": list(range(27, 35+1)),
    "mouth": list(range(48, 60+1)),
    "face": list(range(0, 16+1)),
    "eyebrow_left": list(range(17,21+1)),
    "eyebrow_right": list(range(22,26+1))
}

faceRegions["eyes"] = faceRegions["eye_left"] + faceRegions["eye_right"]
faceRegions["eyebrows"] = faceRegions["eyebrow_left"] + faceRegions["eyebrow_right"]

def loadData(dir="train_imgs"):
    data = {"face":{}, "eyebrows":{}, "eyes":{}, "nose":{}, "mouth":{}}
    
    tc = 0
    
    for region_name, v in data.items():

        paths = os.path.join(dir, region_name, '*/*.*')

        rc = 0

        for path in glob(paths):
            _, feature_name = os.path.split(os.path.dirname(path))
            
            feature_name = feature_name.encode()

            if feature_name not in v:
                v[feature_name] = []

            img = cv2.imread(path)

            if img is None:
                continue
            
            points = getNormalizedFeature(region_name, feature_name, img)

            #skip if no face detected
            if points is not None:
                v[feature_name].append(points)
                rc+=1
                tc+=1
                sys.stdout.write("\033[K")
                print("loading... %s  %d/%d" % (region_name, rc, tc), end="\r")

        print("")
            
    print("loading... Done")
    return data



def getNormalizedFeature(region_name, feature_name, img):
    img = imutils.resize(img, width=800)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    if len(rects) == 0:	#no face was detected
        #sys.exit("No face is detected in %s of %s" % (feature_name, region_name))
        return None
    else:
        faceImg = fa.align(img, gray, rects[0])
        full_rect = dlib.rectangle(0, 0, faceImg.shape[1], faceImg.shape[0])
        shape = predictor(faceImg, full_rect)
        if USE_REGION:
            shape = face_utils.shape_to_np(shape)[faceRegions[region_name]]
        else:
            shape = face_utils.shape_to_np(shape)
        return shape

def getNormalizedFeatures(img, display=False):
    img = imutils.resize(img, width=800)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    data = {"face":[], "eyebrows":[], "eyes":[], "nose":[], "mouth":[]}

    rects = detector(gray, 0)

    if len(rects) == 0:	#no face was detected
        sys.exit("No face is detected")
        return None
    else:
        faceImg = fa.align(img, gray, rects[0])
        full_rect = dlib.rectangle(0, 0, faceImg.shape[1], faceImg.shape[0])
        points = predictor(faceImg, full_rect)
        points = face_utils.shape_to_np(points)
        
        if display:
            cv2.imshow("face", faceImg)
            cv2.waitKey()
        
        for key in data:
            if USE_REGION:
                data[key] = points[faceRegions[key]]
            else:
                data[key] = points
        return faceImg, data