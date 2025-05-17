# Road_Hazard_Detection_Using_Semantic_Segmentation
# Project Objective
The primary objective of the project is to implement semantic segmentation in road scenarios based on the IDD dataset and DeepLabV3+. The project entails investigating the structure of the dataset, particularly the 26 fine-grained level-3 classes for Indian roads. Official AutoNUE tools were employed to generate precise pixel-level masks for training. The DeepLabV3+ model was trained to manage complicated scenes with object boundary preservation. We assessed its performance by the mIoU method at 720p resolution based on AutoNUE standards. The findings assist in evaluating how effectively the model performs in unstructured traffic and indicate its capability to enhance autonomous driving and smart transport systems.
# Dataset
Name: Indian Driving Dataset - Segmentation Part 2

Type: Road scene dataset for semantic segmentation

Size: 5.79 GB

Structure: Images are divided into train, val, and test folders and corresponding segmentation masks are provided for training and validation.

Source: https://www.kaggle.com/datasets/sovitrath/indian-driving-dataset-segmentation-part-2

# Implementation Details
## 1. Setting Up the Environment
We used Windows 11 with Python 3.10.2 to build the project.
We began the implementation on Google Colab because it's an excellent cloud environment for running deep learning models. We enabled GPU acceleration to speed up the training process. In order to keep things organized and ensure that we didn't lose work between sessions, we also mounted Google Drive. There, we placed the dataset and model checkpoints so that if necessary, we could pick up where we left off.

## 2. Preparing the Dataset
We employed the IDD Segmentation Part 2 dataset on Kaggle that contains urban road images and corresponding grayscale segmentation masks. Every pixel in the mask corresponds to a particular object such as a road, car, person etc. . We structured everything into folders and utilized a custom data loader to ensure that every image was paired with the correct mask.
## 3. Creating Segmentation Labels
All images and masks were resized to 256×256 for consistency. The mask values were kept unchanged to preserve labels, and we used data augmentation to mimic different lighting and scene conditions.

● Resize all input images and masks to 256x256 resolution.

● Normalize pixel values (e.g., scale to 0–1 or mean-std normalization).

● Convert masks to class labels (0–25), ensuring correct mapping.
## 4. Using DeepLabV3+ Model
We used DeepLabV3+ since it's excellent in semantic segmentation.


For the semantic segmentation task, the DeepLabV3+ architecture was chosen due to its strong performance in preserving object boundaries and capturing context at multiple scales. The model utilized a ResNet50 backbone pre-trained on ImageNet.

To suit the IDD dataset, which includes 26 semantic classes , the final classifier layer of the model was modified to produce 26 output channels.
## 5. Training the Model
Split data into training and validation sets.
Applied transformations: resizing, normalization, and tensor conversion

● Loss Function: CrossEntropyLoss with ignore_index=255

● Optimizer: Adam, learning rate = 0.0001

● Platform: Trained on Google Colab GPU

● Epochs: 2
## 6. Prediction and Evaluation
After training, we used the model to predict segmentation maps for the validation images.

The evaluation was conducted on the validation set with all predictions and ground truths resized to 1280×720 resolution , in line with the AutoNUE benchmark format.

● Pixel Accuracy

● Mean IoU These metrics indicate that the model was able to reliably distinguish between various elements of the road scene, such as vehicles, roads, pedestrians, and buildings, even in unstructured and cluttered environments.
## 7. Visualization
Overlaid predicted masks on original images using OpenCV

Helped visually assess model accuracy in complex scenes with poor lighting or heavy traffic
# Result and conclusion

![Screenshot 2025-05-17 131350](https://github.com/user-attachments/assets/acb920eb-30ab-4f07-aa4d-1c0b9bae7a4c)
![Screenshot 2025-05-17 131741](https://github.com/user-attachments/assets/e161f385-f907-4d2b-bb5f-8f964be1be63)
![Screenshot 2025-05-17 131628](https://github.com/user-attachments/assets/537eb82e-4fae-4e54-b3f0-b1e16e28df2b)
![Screenshot 2025-05-17 131801](https://github.com/user-attachments/assets/e31901f7-f586-4283-9a5d-7948ea49e84a)

We applied DeepLabV3+ to do semantic segmentation on road scenes, correctly labeling every pixel into major categories such as roads, cars, and pedestrians. The model had good results, and the entire pipeline from data preparation to prediction was good. This project demonstrates how semantic segmentation can enable safer, smarter AI-powered transportation.



