# MobileNetV2 Model for River Water Level Classification
# Individual implementation file for Overleaf repository

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time

# Enable mixed precision if not already set globally
try:
    if tf.keras.mixed_precision.global_policy().name != 'mixed_float16':
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
except:
    pass


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

    def build_model(self):
        """
        Build MobileNetV2 model with optimized classification head

        Architecture:
        - MobileNetV2 base (frozen initially)
        - GlobalAveragePooling2D
        - Dense(256) + ReLU + Dropout(0.4)
        - Dense(128) + ReLU + Dropout(0.3)
        - Dense(num_classes) + Softmax
        """
        print("🔧 Building MobileNetV2 model...")

        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3),
            alpha=self.alpha
        )

        # Freeze base model initially
        base_model.trainable = False

        # Add lightweight classification head
        inputs = base_model.input
        x = base_model.output

        # Global average pooling
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)

        # Lightweight dense layers for mobile efficiency
        x = Dense(256, activation='relu', name='dense_256')(x)
        x = Dropout(0.4, name='dropout_1')(x)

        x = Dense(128, activation='relu', name='dense_128')(x)
        x = Dropout(0.3, name='dropout_2')(x)

        # Output layer
        outputs = Dense(self.num_classes, activation='softmax', name='predictions')(x)

        # Create model
        self.model = Model(inputs, outputs, name='MobileNetV2_WaterLevel')

        # Model summary
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])

        print(f"✅ MobileNetV2 model built successfully!")
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   🔧 Trainable parameters: {trainable_params:,}")
        print(f"   🔒 Frozen parameters: {total_params - trainable_params:,}")
        print(f"   📱 Mobile-optimized: {total_params / 1e6:.2f}M parameters")

        return self.model

    def compile_model(self, learning_rate=0.002):
        """
        Compile the model with mobile-optimized settings
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"✅ Model compiled with Adam optimizer (lr={learning_rate})")

    def setup_data_augmentation(self):
        """
        Setup lightweight data augmentation for mobile deployment
        """
        # Lighter augmentation for faster training
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            zoom_range=0.15,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            rescale=1. / 255
        )

        val_datagen = ImageDataGenerator(rescale=1. / 255)
        return train_datagen, val_datagen

    def setup_callbacks(self, model_save_path='mobilenetv2_best.h5'):
        """
        Setup training callbacks for MobileNetV2
        """
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=12,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=6,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        return callbacks

    def train(self, X_train, y_train, X_val, y_val, epochs=40, batch_size=64):
        """
        Train MobileNetV2 with mobile-optimized settings
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print(f"🚀 Starting MobileNetV2 training...")
        print(f"   📊 Training samples: {len(X_train)}")
        print(f"   🔍 Validation samples: {len(X_val)}")
        print(f"   📦 Batch size: {batch_size}")
        print(f"   🔄 Max epochs: {epochs}")

        # Setup data augmentation
        train_datagen, val_datagen = self.setup_data_augmentation()
        callbacks = self.setup_callbacks()

        # Calculate steps
        steps_per_epoch = max(1, len(X_train) // batch_size)
        validation_steps = max(1, len(X_val) // batch_size)

        # Train the model
        start_time = time.time()
        self.history = self.model.fit(
            train_datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_datagen.flow(X_val, y_val, batch_size=batch_size),
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )

        training_time = time.time() - start_time
        print(f"✅ MobileNetV2 training completed in {training_time / 60:.1f} minutes!")
        return self.history

    def evaluate(self, X_test, y_test, class_names):
        """
        Evaluate the trained MobileNetV2 model
        """
        if self.model is None:
            raise ValueError("Model not built or trained")

        print(f"📊 Evaluating MobileNetV2 model...")

        # Make predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)

        print(f"🎯 MobileNetV2 Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        # Detailed classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

        return accuracy, y_pred, y_pred_proba

    def save_model(self, filepath='mobilenetv2_water_level.h5'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(filepath)
        print(f"💾 MobileNetV2 model saved to {filepath}")


if __name__ == "__main__":
    print("📱 MobileNetV2 Model for River Water Level Classification")
    print("This is a standalone implementation file.")
    print("Import this module and use MobileNetV2Classifier class for training.")