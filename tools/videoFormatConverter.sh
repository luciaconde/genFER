#!/bin/bash

VIDEONAME="$1"
LOCATION="$2"
ORIGFORMAT="$3"
ffmpeg -i "$LOCATION$VIDEONAME.$ORIGFORMAT" "$LOCATION$VIDEONAME.mp4"
