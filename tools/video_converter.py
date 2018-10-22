import os

# Get the list of names of the video files
videos_path = "data/videos/mmi_adjusted/"
videosList = os.listdir(videos_path) # Lists all files (and directories) in the folder

for video in videosList:
    if os.path.isfile(os.path.join(videos_path, video)): # Considers files only
        # Convert video
        video_name = os.path.splitext(video)[0]
        os.system('./videoFormatConverter.sh '+video_name+' '+videos_path+' avi')
        
