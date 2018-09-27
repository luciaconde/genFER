# genFER
CNN-based pipeline for generalized facial expression recognition

File structure of the current version of the project:
```
genFER
│   README.md
│   faceDetectorExtraction.sh
│   video_processor.py
│   cnn_trainer.py
│   dataset.py
│   cnn_tester.py
│   predict_video.py
│
└───data
│   └───test_videos
│   |   │   example_video_for_testing.mp4
│   │
│   └───classes
│   |   └───facial_expression1
│   |   └───facial_expression2
│   |   │   ...
|   |   └───facial_expressionX
|   |       |   face_image1.bmp
|   |       |   face_image2.bmp
|   |       |   ...
|   |       |   face_imageY.bmp
|   |
│   └───videos
|   |   |   training_video1.mp4
|   |   |   training_video2.mp4
|   |   |   ...
|   |   |   training_videoZ.mp4
|   |
|   └───annotations
|   |   |   training_video1_annot.csv
|   |   |   training_video2_annot.csv
|   |   |   ...
|   |   |   training_videoZ_annot.csv
|
└───notes
    │   Instructions.txt
    │   Results_example.txt
```
