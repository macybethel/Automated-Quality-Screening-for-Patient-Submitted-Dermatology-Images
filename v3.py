# Automated Image Processing
# Ohio State Dermatology
#
# Author: Macy Bethel
# Date: February 5, 2026
# Version: 4

import cv2
import os
import shutil
import numpy as np
from ultralytics import YOLO

# -------------------------------
# Quality thresholds (tunable)
# -------------------------------
BLUR_THRESHOLD = 5.0
OVEREXPOSED_THRESHOLD = 0.50
UNDEREXPOSED_THRESHOLD = 0.15
SATURATION_THRESHOLD = 0.15

# -------------------------------
# Quality check functions
# -------------------------------

def blur_score(gray_img):
    """Higher = sharper"""
    return cv2.Laplacian(gray_img, cv2.CV_64F).var()

def exposure_metrics(gray_img):
    dark_pct = np.mean(gray_img < 20)
    bright_pct = np.mean(gray_img > 235)

    return dark_pct, bright_pct

def saturation_metric(rgb_img):
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    return np.mean(saturation > 250)

# -------------------------------
# Main processing function
# -------------------------------

def image_processing(source_folder, approved_folder, rejected_folder):

    os.makedirs(approved_folder, exist_ok=True)
    os.makedirs(rejected_folder, exist_ok=True)

    file_extensions = ('.jpg', '.jpeg', '.png')
    files = [f for f in os.listdir(source_folder) if f.lower().endswith(file_extensions)]

    if not files:
        print("No images found in the source folder.")
        return

    print(f"Found {len(files)} images. Processing...\n")
    num_accepted = 0
    num_rejected = 0
    num_blurry = 0
    num_over = 0
    num_under = 0
    reject_valid = 0
    reject_false = 0
    accept_valid = 0
    accept_false = 0

    for filename in files:
        source_path = os.path.join(source_folder, filename)

        img = cv2.imread(source_path)
        if img is None:
            print(f"{filename}: could not read image")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ---- Quality checks ----
        blur = blur_score(gray)
        dark_pct, bright_pct = exposure_metrics(gray)
        sat_pct = saturation_metric(img)

        # ---- Decision logic ----
        reject_reason = None

        if blur < BLUR_THRESHOLD:
            reject_reason = "blurry / out of focus"
            num_blurry += 1
        elif bright_pct > OVEREXPOSED_THRESHOLD:
            reject_reason = "overexposed"
            num_over +=1
        elif dark_pct > UNDEREXPOSED_THRESHOLD:
            reject_reason = "underexposed"
            num_under += 1
        elif sat_pct > SATURATION_THRESHOLD:
            reject_reason = "oversaturated"

        # ---- Move file ----
        if reject_reason:
            destination = os.path.join(rejected_folder, filename)
            #shutil.move(source_path, destination)
           # print(f"REJECTED: {filename} → {reject_reason}")
            num_rejected +=1
            if "Slide" in filename:
                reject_valid += 1
            else:
                reject_false += 1
        else:
            destination = os.path.join(approved_folder, filename)
            #shutil.move(source_path, destination)
          #  print(f"ACCEPTED: {filename}")
            num_accepted +=1
            if "Slide" in filename:
                accept_false += 1
            else:
                accept_valid += 1

    print("\nAll images processed.")
    return num_rejected, num_accepted, num_blurry, num_over, num_under, reject_valid, reject_false, accept_valid, accept_false

# -------------------------------
# Settings
# -------------------------------

source_folder = "Skin Images"
approved_folder = "Accepted_Images"
rejected_folder = "Rejected_Images"



num_rejected, num_accepted, num_blurry, num_over, num_under,  reject_valid, reject_false, accept_valid, accept_false= image_processing(source_folder, approved_folder, rejected_folder)

print(f"Number accepted: {num_accepted}")
print(f"Number rejected: {num_rejected}")
print(f"Number blurry: {num_blurry}")
print(f"Number Overexposed: {num_over}")
print(f"Number Underexposed: {num_under}")

print(f"{reject_valid}")
print(f"{reject_false}")
print(f"{accept_valid}")
print(f"{accept_false}")


