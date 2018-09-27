# genFER
CNN-based pipeline for generalized facial expression recognition

#### File structure of project (current version)
```
genFER
│   README.md
│   faceDetectorExtraction.sh: calls OpenFace's CLI option for feature extraction
│   video_processor.py: processes the extracted face images and orders them for training
│   cnn_trainer.py: trains the convolutional neural network while performing k-fold cross-validation
│   dataset.py: loads and processes the stored face images for training
│   cnn_tester.py: tests (during training) the current model of the convolutional neural network
│   predict_video.py: performs a per-frame facial expression classification of a specific video
│
└───data
│   └───test_videos: contains the video(s) to be classified (using predict_video.py)
│   |   │   example_video_for_testing.mp4
│   │
│   └───classes: contains the extracted face images ordered per annotated facial expression
│   |   └───facial_expression1
│   |   └───facial_expression2
│   |   │   ...
|   |   └───facial_expressionX
|   |       |   face_image1.bmp
|   |       |   face_image2.bmp
|   |       |   ...
|   |       |   face_imageY.bmp
|   |
│   └───videos: contains the videos to be used during training
|       |   training_video1.mp4
|       |   training_video2.mp4
|       |   ...
|       |   training_videoZ.mp4
|       |
|       └───annotations: contains the (manually tagged) annotations for the training videos
|           |   training_video1_annot.csv
|           |   training_video2_annot.csv
|           |   ...
|           |   training_videoZ_annot.csv
|
└───notes
    │   Instructions.txt
    │   Results_example.txt
```
