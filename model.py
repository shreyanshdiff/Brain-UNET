from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation
import tensorflow as tf


def build_unet(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, (3, 3), padding="same")(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), padding="same")(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), padding="same")(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = Conv2D(512, (3, 3), padding="same")(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)

    # Decoder
    up1 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(conv4)
    up1 = concatenate([up1, conv3])
    conv5 = Conv2D(256, (3, 3), padding="same")(up1)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)

    up2 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(conv5)
    up2 = concatenate([up2, conv2])
    conv6 = Conv2D(128, (3, 3), padding="same")(up2)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6)

    up3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(conv6)
    up3 = concatenate([up3, conv1])
    conv7 = Conv2D(64, (3, 3), padding="same")(up3)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7)

    # Output layer
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(conv7)

    return Model(inputs, outputs)
