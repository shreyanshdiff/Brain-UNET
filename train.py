from model import build_unet
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from keras_preprocessing.image import ImageDataGenerator


def dice_loss(Y_true, Y_pred):
    numerator = 2 * tf.reduce_sum(Y_true * Y_pred)
    denominator = tf.reduce_sum(Y_true + Y_pred)  # Fixed
    return 1 - (numerator / (denominator + tf.keras.backend.epsilon()))  # Prevent division by zero


def compute_sample_weights(Y_train):
    flat_labels = Y_train.flatten()
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(flat_labels),
        y=flat_labels
    )
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}

    sample_weights = np.vectorize(class_weights_dict.get)(Y_train)
    return sample_weights.reshape(Y_train.shape)  # Reshape to match Y_train


class WeightedDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, Y, batch_size=16, augment=False):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.augment = augment
        self.datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        ) if augment else None

        # ✅ Compute sample weights
        self.sample_weights = compute_sample_weights(Y)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        batch_X = self.X[index * self.batch_size:(index + 1) * self.batch_size]
        batch_Y = self.Y[index * self.batch_size:(index + 1) * self.batch_size]
        batch_weights = self.sample_weights[index * self.batch_size:(index + 1) * self.batch_size]

        if self.augment:
            batch_X, batch_Y = next(self.datagen.flow(batch_X, batch_Y, batch_size=self.batch_size))

        return batch_X, batch_Y, batch_weights  # ✅ Include sample weights


def train_model(X_train, Y_train, X_val, Y_val):
    model = build_unet()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=dice_loss,
                  metrics=["accuracy"])
    
    print("[INFO] Training U-Net model...")

    # ✅ Use the custom data generator instead of ImageDataGenerator
    train_gen = WeightedDataGenerator(X_train, Y_train, batch_size=16, augment=True)
    val_gen = WeightedDataGenerator(X_val, Y_val, batch_size=16, augment=False)

    # ✅ Now model.fit will work without errors
    model.fit(train_gen, epochs=30, validation_data=val_gen)

    # Save the model
    model.save("brain_tumor_unetV2.h5")
    print("[INFO] Model saved!")

    return model
