# this code train the svm for classification

from utils import *
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from svmutil import *

SAVE_PATH="data/trained_svms.pkl"
SAVE_TRAIN_DATA_PATH = "data/train_data.pkl"
LIBSVM_SVMS_PATH = "data/%s.svm"
LIBSVM_LABELS_PATH = "data/labels.txt"

GET_CROSS_VAL = False  
IS_BUILD_LIBSVM_MODEL = False   # save as libsvm model for prediction in android

if os.path.isfile(SAVE_TRAIN_DATA_PATH):
    data = joblib.load(SAVE_TRAIN_DATA_PATH)
else:
    data = loadData()
    joblib.dump(data, SAVE_TRAIN_DATA_PATH)

svms = {}

if IS_BUILD_LIBSVM_MODEL:
    labels_file = open(LIBSVM_LABELS_PATH, 'w')

for region_name, features in data.items():
    print("training svm for %s      "% (region_name))


    # split the data into training set and test set
    
    if not IS_BUILD_LIBSVM_MODEL:
                
        X = [] 
        y = []
        for feature_name, feature_shapes in features.items():
            for shape in feature_shapes:
                X.append(shape.flatten())
                y.append(feature_name)
            
        X = np.squeeze(np.array(X))
        y = np.array(y,dtype='S128')

        # split data
        # X_train, X_test, y_train, y_test = train_test_split(X,y)

        svms[region_name.encode()] = svm.SVC(kernel="linear", probability=True)

        if GET_CROSS_VAL:
            scores = cross_val_score(svms[region_name.encode()], X, y, cv=5)
            print("Cross val score: ", scores)
            print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        # train for deployment
        svms[region_name.encode()].fit(X, y)

    else:
        X = [] 
        y = []
        for i, (feature_name, feature_shapes) in enumerate(features.items()):
            for shape in feature_shapes:
                X.append(shape.flatten())
                y.append(i)
            
        X = np.squeeze(np.array(X))
        y = np.array(y,dtype='uint8')

        labels_file.write("%s\n" % region_name)
        labels_file.write(LIBSVM_SVMS_PATH % region_name)
        labels_file.write(" ")
        labels_file.write(" ".join([k.decode() for k in features.keys()]))
        labels_file.write("\n")

        # train for libsvm
        prob = svm_problem(y.tolist(), X.tolist())

        param = svm_parameter("-h 0 -s 0 -t 1 -b 1")
        m=svm_train(prob, param)
        svm_save_model(LIBSVM_SVMS_PATH % region_name, m)

if IS_BUILD_LIBSVM_MODEL:
    labels_file.close()
    
print("training svm... Done") 


joblib.dump(svms, SAVE_PATH)
print("svm saved!")