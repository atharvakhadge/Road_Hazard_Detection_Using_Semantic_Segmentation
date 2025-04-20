import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from deeplabv3_model import deeplabv3_model  # DeepLabV3+ model import karenge

# Data load function
def load_data(img_dir, mask_dir):
    # Ensure that the paths are correct for loading npy files
    X = np.load(img_dir)  
    y = np.load(mask_dir) 
    return X, y

# Model define karo
model = deeplabv3_model(input_shape=(256, 256, 3), num_classes=34)  # Input size aur num_classes specify kar rahe hain
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Multi-class segmentation ke liye sparse_categorical_crossentropy
              metrics=['accuracy'])

# Ensure paths for the train images and masks
X_train = np.load('C:/Users/anush/Desktop/IDD-RoadSceneSegmentation/preprocessed_images/train/X_train.npy')  # Training images
y_train = np.load('C:/Users/anush/Desktop/IDD-RoadSceneSegmentation/preprocessed_masks/train/y_train.npy')  # Training masks

# ModelCheckpoint se best model ko save karna
checkpoint = ModelCheckpoint("best_deeplab_model.h5", save_best_only=True, monitor='val_loss', mode='min')

# Model ko train karo
model.fit(X_train, y_train,  # Training data
          batch_size=8,  # Batch size
          epochs=50,  # Epochs
          validation_split=0.1,  # 10% validation data
          callbacks=[checkpoint])  # Checkpoint to save the best model

# Agar validation data available ho to evaluate bhi kar sakte ho
# X_val = np.load('X_val.npy')
# y_val = np.load('y_val.npy')
# loss, accuracy = model.evaluate(X_val, y_val)
# print(f"Validation Loss: {loss}")
# print(f"Validation Accuracy: {accuracy}")
