from __future__ import with_statement
from google_images_download import google_images_download
import json, os

TRAIN_DATA_DIR_PATH = "train_imgs"

#load the analysis for naming reference
with open('data/analysis.json') as f:
    analysis = json.load(f)

global_args = {
    "limit":75,
    "output_directory":TRAIN_DATA_DIR_PATH,
    "prefix":"",
    "keywords":"",
    "prefix_keywords":"面相"
    
}

for region in analysis["face_regions"]:
    
    region_name = region["name"]

    for feature in region["features"]:

        download_args = global_args
        download_args["output_directory"] = os.path.join(TRAIN_DATA_DIR_PATH, region_name)
        download_args["keywords"] = feature["name"]

        response = google_images_download.googleimagesdownload()
        response.download(download_args)

        default_fking_ugly_dirname = os.path.join(TRAIN_DATA_DIR_PATH, region_name, download_args["prefix_keywords"] + " " + feature["name"])
        os.rename(default_fking_ugly_dirname,default_fking_ugly_dirname.replace( download_args["prefix_keywords"] + " ", ""))


