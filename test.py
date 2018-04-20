# test the training data

from utils import *
from sklearn import svm
from sklearn.externals import joblib
import numpy as np
import cv2
import argparse
import json
from textwrap import fill

from imutils.convenience import url_to_image 

TEST_IMAGE_PATH="test_imgs/test.png"
SAVE_PATH="data/trained_svms.pkl"

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default=None,
	help="input image")
ap.add_argument("-u", "--url", type=str, default=None,
	help="input image url")
args = vars(ap.parse_args())

#load the analysis for naming reference
with open('data/analysis.json') as f:
    analysis = json.load(f)



def apply(img):
    data = getNormalizedFeatures(img, False)

    svms = joblib.load(SAVE_PATH)

    for region_name, points in data.items():
        X = [points.flatten()]

        y = svms[region_name.encode()].predict(X)[0].decode()
        prob = svms[region_name.encode()].predict_proba(X)
        max_prob = np.amax(prob)*100

        print("【 %s 】\t %s %f%%" % (region_name, y, max_prob))

        for region in analysis["face_regions"]:
            if region["name"] == region_name:
                for feature in region["features"]:
                    if feature["name"] == y:
                        print(fill(feature["analysis"], width=18))


        print(" ")


if __name__ == '__main__':

    if args["image"] is not None:
        img = cv2.imread(args["image"])
    elif args["url"] is not None:
        img = url_to_image(args["url"])
    else:
        img = cv2.imread(TEST_IMAGE_PATH)

    apply(img)