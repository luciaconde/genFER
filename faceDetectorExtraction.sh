#!/bin/bash

VIDEONAME="$1"
LOCATION="$2"
/home/lucia/opencv-3.4.0/build/OpenFace/build/bin/FeatureExtraction -f "$LOCATION$VIDEONAME" -out_dir "$LOCATION"

# OpenFace's FeatureExtraction outputs image (.bmp) files of the detected face on every frame, having already performed face alignment and pose normalization
