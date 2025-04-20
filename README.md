# Road_Hazard_Detection_Using_Semantic_Segmentation
# Project Objective
The primary objective of the project is to implement semantic segmentation in road scenarios based on the IDD dataset and DeepLabV3+. The project entails investigating the structure of the dataset, particularly the 26 fine-grained level-3 classes for Indian roads. Official AutoNUE tools were employed to generate precise pixel-level masks for training. The DeepLabV3+ model was trained to manage complicated scenes with object boundary preservation. We assessed its performance by the mIoU method at 720p resolution based on AutoNUE standards. The findings assist in evaluating how effectively the model performs in unstructured traffic and indicate its capability to enhance autonomous driving and smart transport systems.
# Dataset
The dataset used was the IDD-20K dataset, released as part of the AutoNUE Challenge 2021. It contains over 20,000 images of Indian road scenes, annotated at three levels of hierarchy. For this project, Level 3 annotations (26 classes) were used.

IDD-20K Part I
IDD-20K Part II
