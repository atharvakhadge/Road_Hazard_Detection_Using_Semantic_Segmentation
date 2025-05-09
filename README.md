# Road_Hazard_Detection_Using_Semantic_Segmentation
# Project Objective
The primary objective of the project is to implement semantic segmentation in road scenarios based on the IDD dataset and DeepLabV3+. The project entails investigating the structure of the dataset, particularly the 26 fine-grained level-3 classes for Indian roads. Official AutoNUE tools were employed to generate precise pixel-level masks for training. The DeepLabV3+ model was trained to manage complicated scenes with object boundary preservation. We assessed its performance by the mIoU method at 720p resolution based on AutoNUE standards. The findings assist in evaluating how effectively the model performs in unstructured traffic and indicate its capability to enhance autonomous driving and smart transport systems.
# Dataset
The dataset used was the IDD-20K dataset, released as part of the AutoNUE Challenge 2021. It contains over 20,000 images of Indian road scenes, annotated at three levels of hierarchy. For this project, Level 3 annotations (26 classes) were used.

IDD-20K Part I
IDD-20K Part II

Download the dataset from: https://idd.insaan.iiit.ac.in/
# Implementation Details
## 1. Setting Up the Environment
We used Windows 11 with Python 3.10.2 to build the project.


After activating the environment, we installed the necessary Python libraries:


•	numpy


•	pandas==1.2.1


•	tqdm


•	Pillow


•	scipy==1.1.0


•	imageio
## 2. Preparing the Dataset
We used the IDD-20K dataset, which is part of the AutoNUE Challenge 2021. It contains more than 20,000 images of Indian road scenes.
We worked with Level-3 annotations, which provide detailed labeling for 26 different classes like road, car, pedestrian, etc.
We combined Part I and Part II of the dataset into a single folder for easier access.
## 3. Creating Segmentation Labels
The dataset comes with annotations in JSON format, so we needed to convert them into segmentation mask images (PNG format).


•	We downloaded the official AutoNUE tools from GitHub.


•	Each pixel in the output image had a value from 0 to 25, representing one of the 26 classes.
## 4. Using DeepLabV3+ Model
We used DeepLabV3+ since it's excellent in semantic segmentation.


Here's what it does:


• Backbone: We employed a pre-trained ResNet-101 to extract the features from images.


• Encoder: It employs ASPP (Atrous Spatial Pyramid Pooling) to acquire features at diverse scales.


• Decoder: It returns the details to acquire precise object shapes and borders.
## 5. Training the Model
Training settings:


•	Optimizer: Adam


•	Learning Rate: 0.001


•	Batch Size: 8


•	Loss Function: CrossEntropyLoss


•	Epochs: 50
## 6. Prediction and Evaluation
After training, we used the model to predict segmentation maps for the validation images.


•	Each output was a PNG mask showing which pixels belong to which class.


•	We resized both predicted and ground truth masks to 1280x720 using nearest neighbor interpolation.


•	To check how well the model worked, we used the mean Intersection over Union metric.
## 7. Visualization and Results
To visualize the results, we used OpenCV to overlay the predicted masks on the original images.


This helped us:


•	See how accurately the model recognized different objects


•	Understand its performance in difficult scenes like dark lighting or heavy traffic.
# Result and conclusion

![Screenshot 2025-04-20 195104](https://github.com/user-attachments/assets/eaa7b3dc-7a74-492e-99b1-c5f86af84d49)
![Screenshot 2025-04-20 194248](https://github.com/user-attachments/assets/537877f1-9231-44e2-a8dc-3ca5e35cf374)
![Screenshot 2025-04-20 194100](https://github.com/user-attachments/assets/9fe27951-2baf-4b7a-8926-0d130d42f4d0)
![Screenshot 2025-04-20 142315](https://github.com/user-attachments/assets/14b1bf93-4628-4719-9f13-8e45f6a0e871)
![Screenshot 2025-04-20 142656](https://github.com/user-attachments/assets/f25d27a7-d199-49b5-95bd-52cb9ce69925)


We applied DeepLabV3+ to do semantic segmentation on road scenes, correctly labeling every pixel into major categories such as roads, cars, and pedestrians. The model had good results, and the entire pipeline from data preparation to prediction was good. This project demonstrates how semantic segmentation can enable safer, smarter AI-powered transportation.



