import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

def deeplabv3_model(input_shape=(256, 256, 3), num_classes=34):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    layer_names = [
        'conv1_relu',   # 64x64
        'conv2_block3_out',  # 64x64
        'conv3_block4_out',  # 32x32
        'conv4_block6_out',  # 16x16
        'conv5_block3_out'   # 8x8
    ]
    layers_output = [base_model.get_layer(name).output for name in layer_names]
    backbone = models.Model(inputs=base_model.input, outputs=layers_output)

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    features = backbone(x)
    x = features[-1]

    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    x = layers.Conv2D(num_classes, 1, padding='same', activation='softmax')(x)
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)

    return models.Model(inputs=inputs, outputs=x)