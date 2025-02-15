import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set image size
IMG_SIZE = (256, 256)

# Paths to datasets
POSITIVE_PATH = r"data\affected_yes"  # Tumor images
NEGATIVE_PATH = r"data\affected_no"  # No tumor images

def load_images_from_folder(folder, label):
    images, masks = [], []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load grayscale
        img = cv2.resize(img, IMG_SIZE)  # Resize
        img = img / 255.0  # Normalize
        
        # Create mask (1 for tumor, 0 for no tumor)
        mask = np.ones_like(img) if label == 1 else np.zeros_like(img)
        
        # Expand dimensions
        images.append(np.expand_dims(img, axis=-1))
        masks.append(np.expand_dims(mask, axis=-1))

    return np.array(images), np.array(masks)

def preprocess_data():
    print("[INFO] Loading dataset...")
    X_pos, Y_pos = load_images_from_folder(POSITIVE_PATH, label=1)
    X_neg, Y_neg = load_images_from_folder(NEGATIVE_PATH, label=0)

    # Combine dataset
    X = np.concatenate([X_pos, X_neg], axis=0)
    Y = np.concatenate([Y_pos, Y_neg], axis=0)

    # Split into train, validation, test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    print(f"[INFO] Dataset split: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test
