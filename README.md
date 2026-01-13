# Just Walk Out: YOLO-based Fruit Detection
## About This Project
This project implements a YOLO-style object detection model  
from scratch for detecting fruits in images.
## Inspiration
It is inspired by automatic checkout systems such as  
Amazon “Just Walk Out”, where real-time object detection  
is used to identify and track products.
## Goal
The main goal is to understand how YOLO works internally,  
including CNN-based feature extraction, grid-based prediction,  
IoU-based responsibility assignment, and loss optimization.
## Functions
The model takes an RGB image as input, divides it into  
a fixed grid, and predicts bounding boxes, confidence scores,  
and class probabilities in a single forward pass.
## Strength
This single-pass design enables fast, end-to-end detection  
and is suitable for real-time applications.
## Data Source (COCO 2017)
The project follows the COCO 2017 object detection format  
and uses a filtered subset of fruit categories  
(apple, banana, orange).
The dataset is not included in this repository.  
COCO 2017 dataset:  
https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset
## FILES
Fruit Detector AI Project Teams#3.pdf is the content of project showcase.

Yolo mechanism.pdf is the research paper that we based on to implement key functions.

dataset_coco.py is the file processing COCO raw data.

loss_yolo.py implements the loss function.

utils_bbx.py calculates the IoU of two bounding boxes.

model_yolo.py rebuilds the Yolo model.

train.py trains the rebuilt Yolo model.



