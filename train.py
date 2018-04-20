# this code train the svm for classification

from utils import *
from sklearn import svm
from sklearn.externals import joblib
import numpy as np

SAVE_PATH="data/trained_svms.pkl"
SAVE_TRAIN_DATA_PATH = "data/train_data.pkl"

if os.path.isfile(SAVE_TRAIN_DATA_PATH):
    data = joblib.load(SAVE_TRAIN_DATA_PATH)
else:
    data = loadData()
    joblib.dump(data, SAVE_TRAIN_DATA_PATH)

svms = {}

for region_name, features in data.items():
    print("training svm for %s      "% (region_name), end="\r")
    
    X = [] 
    y = []
    for feature_name, feature_shapes in features.items():
        for shape in feature_shapes:
            X.append(shape.flatten())
            y.append(feature_name)
        
    X = np.squeeze(np.array(X))
    y = np.array(y,dtype='S128')
    svms[region_name.encode()] = svm.SVC(kernel="linear", probability=True, class_weight='balanced')
    svms[region_name.encode()].fit(X, y)

print("training svm... Done      ") 


joblib.dump(svms, SAVE_PATH)
print("svm saved!")