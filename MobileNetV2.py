# MobileNetV2 Model for River Water Level Classification
# Individual implementation file for Overleaf repository

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, DepthwiseConv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time


class MobileNetV2Classifier:
    """
    MobileNetV2-based classifier for river water level prediction

    Architecture Features:
    - Pre-trained MobileNetV2 backbone (ImageNet weights)
    - Depthwise separable convolutions for efficiency
    - Inverted residual blocks with linear bottlenecks
    - Lightweight design optimized for mobile/edge deployment
    - Custom classification head with minimal parameters
    """

    def __init__(self, num_classes, img_size=(224, 224), alpha=1.0):
        self.num_classes = num_classes
        self.img_size = img_size
        self.alpha = alpha  # Width multiplier for MobileNetV2
        self.model = None
        self.history = None

    def build_model(self): #Build MobileNetV2 model with optimized classification