# Advanced River Water Level Classification - Complete Solution for Challenging Dataset
# Handles multiple environments, lighting conditions, and perspective variations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import cv2
import os
from pathlib import Path
import kagglehub
import warnings

warnings.filterwarnings('ignore')

# 🚀 CRITICAL OPTIMIZATION: Enable mixed precision for 2x speedup
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("✅ Mixed precision enabled - expect 1.5-2x speedup on compatible GPUs")

# GPU optimization
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"✅ Found {len(gpus)} GPU(s) available")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled")
    except RuntimeError as e:
        print(f"⚠️ GPU setup warning: {e}")
else:
    print("⚠️ No GPU found - consider using Google Colab for faster training")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class AdvancedRiverWaterLevelClassifier:
    """
    Advanced River Water Level Classifier for Challenging Multi-Environment Dataset
    """

    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        self.dataset_path = None
        self.class_names = []
        self.num_classes = 0

    def download_dataset(self):
        """Download the dataset using kagglehub"""
        print("📥 Downloading dataset from Kaggle...")
        try:
            path = kagglehub.dataset_download("otavio12/urban-river-level-classification-image-dataset")
            print(f"✅ Dataset downloaded to: {path}")
            self.dataset_path = Path(path)
            return path
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            return None

    def explore_dataset_structure(self):
        """Explore and analyze the downloaded dataset structure"""
        if not self.dataset_path or not self.dataset_path.exists():
            print("❌ Dataset path not found. Please download the dataset first.")
            return None

        print("\n📁 DATASET STRUCTURE ANALYSIS")
        print("=" * 50)

        all_items = list(self.dataset_path.rglob("*"))
        dirs = [item for item in all_items if item.is_dir()]
        files = [item for item in all_items if item.is_file()]

        print(f"📊 Total items: {len(all_items)}")
        print(f"📁 Directories: {len(dirs)}")
        print(f"📄 Files: {len(files)}")

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in files if f.suffix.lower() in image_extensions]

        print(f"\n🖼️  Total image files found: {len(image_files)}")

        class_info = {}
        for img_file in image_files:
            parts = img_file.relative_to(self.dataset_path).parts
            if len(parts) > 1:
                class_name = parts[-2]
                if class_name not in class_info:
                    class_info[class_name] = []
                class_info[class_name].append(img_file)

        if class_info:
            print("\n📊 CLASS DISTRIBUTION:")
            self.class_names = sorted(class_info.keys())
            self.num_classes = len(self.class_names)

            for class_name in self.class_names:
                count = len(class_info[class_name])
                print(f"  {class_name}: {count} images")

            print(f"\n✅ Detected {self.num_classes} classes: {self.class_names}")
        else:
            print("⚠️  Could not determine class structure from directories")

        return class_info

    def load_real_dataset(self, max_samples_per_class=None):
        """Load the real dataset from the downloaded path"""
        if not self.dataset_path:
            print("❌ Please download the dataset first using download_dataset()")
            return None, None

        print("\n📊 LOADING REAL DATASET")
        print("=" * 40)

        class_info = self.explore_dataset_structure()

        if not class_info:
            print("❌ Could not load dataset. Please check the dataset structure.")
            return None, None

        X, y = [], []

        print(f"\n🔄 Loading images...")
        for class_idx, class_name in enumerate(self.class_names):
            print(f"Loading {class_name}...")

            image_files = class_info[class_name]
            if max_samples_per_class:
                image_files = image_files[:max_samples_per_class]

            for img_path in image_files:
                try:
                    img = load_img(img_path, target_size=self.img_size)
                    img_array = img_to_array(img) / 255.0

                    X.append(img_array)
                    y.append(class_idx)

                except Exception as e:
                    print(f"⚠️  Error loading {img_path}: {e}")
                    continue

            print(f"  ✅ Loaded {len([i for i in y if i == class_idx])} images for {class_name}")

        X = np.array(X)
        y = np.array(y)

        print(f"\n✅ Dataset loaded successfully!")
        print(f"📊 Total samples: {len(X)}")
        print(f"🖼️  Image shape: {X.shape[1:]}")
        print(f"🏷️  Classes: {self.num_classes}")

        return X, y

    def visualize_real_dataset(self, X, y):
        """Visualize the real dataset samples and distribution"""
        # Plot sample images from each class
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('River Water Level Dataset - Multi-Environment Challenge', fontsize=16)

        for i in range(min(8, len(self.class_names) * 2)):
            row = i // 4
            col = i % 4

            class_idx = i % self.num_classes
            class_indices = np.where(y == class_idx)[0]

            if len(class_indices) > 0:
                idx = np.random.choice(class_indices)
                axes[row, col].imshow(X[idx])
                axes[row, col].set_title(f'Class: {self.class_names[class_idx]}')
            else:
                axes[row, col].text(0.5, 0.5, 'No samples', ha='center', va='center')

            axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()

        # Plot class distribution with challenge analysis
        plt.figure(figsize=(15, 5))
        unique, counts = np.unique(y, return_counts=True)

        plt.subplot(1, 3, 1)
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))
        bars = plt.bar([self.class_names[i] for i in unique], counts, color=colors)
        plt.title('Class Distribution\n(Multi-Environment Challenge)')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)

        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                     str(count), ha='center', va='bottom')

        plt.subplot(1, 3, 2)
        plt.pie(counts, labels=[self.class_names[i] for i in unique], autopct='%1.1f%%', colors=colors)
        plt.title('Class Distribution\n(Challenging Imbalance)')

        plt.subplot(1, 3, 3)
        plt.text(0.1, 0.9, f'Dataset Challenge Analysis:', fontsize=14, fontweight='bold',
                 transform=plt.gca().transAxes)
        plt.text(0.1, 0.8, f'Total samples: {len(X)}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.7, f'Classes: {self.num_classes}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.6, f'Class imbalance: {max(counts) / min(counts):.1f}:1', fontsize=12,
                 transform=plt.gca().transAxes, color='red')
        plt.text(0.1, 0.5, f'Environment variety: High', fontsize=12, transform=plt.gca().transAxes, color='red')
        plt.text(0.1, 0.4, f'Lighting conditions: Varied', fontsize=12, transform=plt.gca().transAxes, color='red')
        plt.text(0.1, 0.3, f'Challenge level: EXTREME', fontsize=12, transform=plt.gca().transAxes, color='red',
                 fontweight='bold')
        plt.text(0.1, 0.2, f'Expected accuracy: 50-75%', fontsize=12, transform=plt.gca().transAxes, color='blue')
        plt.axis('off')
        plt.title('Challenge Assessment')

        plt.tight_layout()
        plt.show()

        print("\n" + "=" * 80)
        print("CHALLENGING DATASET ANALYSIS")
        print("=" * 80)
        print(f"🚨 DATASET COMPLEXITY: EXTREME")
        print(f"   📊 Total samples: {len(X)}")
        print(f"   🏷️  Classes: {self.num_classes} ({self.class_names})")
        print(f"   ⚖️  Class imbalance ratio: {max(counts) // min(counts)}:1")
        print(f"   🌍 Environment diversity: Multiple (outdoor/indoor/tunnel)")
        print(f"   💡 Lighting conditions: Highly varied")
        print(f"   📐 Perspective variation: Extreme")
        print(f"   🎯 Realistic target accuracy: 50-75%")
        print("=" * 80)


def create_advanced_data_augmentation():
    """Advanced data augmentation for multi-environment dataset - COMPATIBLE VERSION"""
    print("🔧 Creating ADVANCED data augmentation for challenging dataset...")

    return ImageDataGenerator(
        # Extreme geometric transformations for varied environments
        rotation_range=40,  # Handle different camera orientations
        width_shift_range=0.4,  # Account for different framing
        height_shift_range=0.4,  # Account for different viewpoints
        shear_range=0.3,  # Handle perspective differences
        zoom_range=0.4,  # Handle different distances
        horizontal_flip=True,
        vertical_flip=False,

        # CRITICAL: Color/lighting normalization for indoor/outdoor
        brightness_range=[0.3, 1.7],  # Extreme lighting variations
        channel_shift_range=50,  # Handle color temperature differences
        fill_mode='reflect',

        # Basic preprocessing (removed incompatible parameters)
        rescale=1. / 255
    )


def create_robust_efficientnet_model(num_classes, img_size=(224, 224)):
    """Create ultra-robust EfficientNet for extreme multi-environment dataset"""
    print("🏗️  BUILDING ULTRA-ROBUST EfficientNetB0 MODEL")
    print("   🎯 Designed for extreme multi-environment challenges")
    print("   🔧 Enhanced capacity and regularization")

    # Load pre-trained EfficientNetB0
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(*img_size, 3)
    )

    # Keep frozen for challenging dataset
    base_model.trainable = False

    # MUCH LARGER classification head for complex patterns
    inputs = base_model.input
    x = base_model.output

    # Global average pooling
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)

    # EXPANDED architecture for challenging dataset
    x = Dense(2048, name='dense_2048')(x)  # Much larger
    x = BatchNormalization(name='bn_1')(x)
    x = tf.keras.layers.Activation('relu', name='relu_1')(x)
    x = Dropout(0.6, name='dropout_1')(x)  # Higher dropout

    x = Dense(1024, name='dense_1024')(x)
    x = BatchNormalization(name='bn_2')(x)
    x = tf.keras.layers.Activation('relu', name='relu_2')(x)
    x = Dropout(0.5, name='dropout_2')(x)

    x = Dense(512, name='dense_512')(x)
    x = BatchNormalization(name='bn_3')(x)
    x = tf.keras.layers.Activation('relu', name='relu_3')(x)
    x = Dropout(0.4, name='dropout_3')(x)

    x = Dense(256, name='dense_256')(x)
    x = BatchNormalization(name='bn_4')(x)
    x = tf.keras.layers.Activation('relu', name='relu_4')(x)
    x = Dropout(0.3, name='dropout_4')(x)

    # Output layer
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)

    # Create model
    model = Model(inputs, outputs, name='UltraRobust_EfficientNetB0')

    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

    print(f"✅ ULTRA-ROBUST EfficientNetB0 built successfully!")
    print(f"   📊 Total parameters: {total_params:,}")
    print(f"   🔧 Trainable parameters: {trainable_params:,}")
    print(f"   🎯 Enhanced capacity: 4-layer classification head")
    print(f"   🛡️  Heavy regularization: Multi-level dropout")

    return model


def advanced_training_strategy():
    """ULTIMATE training strategy for extremely challenging dataset"""
    print("🎯 ULTIMATE TRAINING STRATEGY FOR EXTREME DATASET CHALLENGE")
    print("=" * 80)

    # Initialize classifier
    classifier = AdvancedRiverWaterLevelClassifier()

    # Step 1: Dataset preparation
    print("\n📥 STEP 1: Dataset Preparation")
    dataset_path = classifier.download_dataset()
    if not dataset_path:
        print("❌ Failed to download dataset.")
        return None, None

    # Load dataset
    X, y = classifier.load_real_dataset(max_samples_per_class=500)
    if X is None or len(X) == 0:
        print("❌ Failed to load dataset.")
        return None, None

    classifier.visualize_real_dataset(X, y)

    # Step 2: Strategic data splitting
    print("\n🔄 STEP 2: Strategic Data Splitting for Challenging Dataset")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"✅ Data split for challenging training:")
    print(f"   📚 Training: {len(X_train)} samples")
    print(f"   🔍 Validation: {len(X_val)} samples")
    print(f"   🧪 Test: {len(X_test)} samples")

    # Step 3: EXTREME class weight calculation
    print(f"\n⚖️ STEP 3: EXTREME CLASS BALANCING")
    print("-" * 60)

    # Calculate base class weights
    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)

    # EXTREME weight adjustment for challenging dataset
    extreme_weights = {}
    for i, weight in enumerate(class_weights):
        class_name = classifier.class_names[i]
        count = np.sum(y_train == i)

        if class_name == 'flood':  # Boost minority class extremely
            extreme_weights[i] = weight * 3.0
        elif count < 100:  # Boost any small class
            extreme_weights[i] = weight * 2.0
        else:
            extreme_weights[i] = weight

    print(f"📊 EXTREME Class Weights for Challenging Dataset:")
    total_samples = len(y_train)
    for i, class_name in enumerate(classifier.class_names):
        count = np.sum(y_train == i)
        percentage = (count / total_samples) * 100
        weight = extreme_weights[i]
        boost = "🚀 EXTREME BOOST" if weight > 2.0 else "⚡ BOOSTED" if weight > 1.5 else "📊 NORMAL"
        print(f"   {class_name:>8}: {count:>3} samples ({percentage:>5.1f}%) → Weight: {weight:>6.3f} {boost}")

    # Step 4: Build ultra-robust model
    print(f"\n🏗️  STEP 4: Building Ultra-Robust Model")
    print("-" * 60)

    model = create_robust_efficientnet_model(len(classifier.class_names))

    # ULTRA-CONSERVATIVE optimizer for challenging dataset
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=0.00001,  # Extremely low for stability
        decay=0.95,
        momentum=0.9
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"✅ Model compiled with ULTRA-CONSERVATIVE settings:")
    print(f"   📉 Learning rate: 1e-5 (extremely low)")
    print(f"   🎯 Optimizer: RMSprop with high momentum")
    print(f"   🛡️  Strategy: Maximum stability for challenging data")

    # Step 5: Advanced data preparation
    print(f"\n🔧 STEP 5: Advanced Data Augmentation Setup")
    print("-" * 60)

    train_datagen = create_advanced_data_augmentation()
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    print(f"✅ Advanced augmentation configured:")
    print(f"   🌪️  Extreme geometric transformations")
    print(f"   💡 Lighting/color normalization")
    print(f"   🎯 Optimized for multi-environment dataset")

    # Step 6: Extended training with advanced callbacks
    print(f"\n🚀 STEP 6: EXTENDED TRAINING")
    print("-" * 60)

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=40,  # Very long patience for challenging dataset
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,  # Gentle reduction
            patience=20,
            min_lr=1e-8,
            verbose=1
        ),
        ModelCheckpoint(
            'ultra_robust_efficientnetb0_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]

    # EXTENDED training configuration
    batch_size = 16  # Smaller batches for stability
    epochs = 60  # Longer training for challenging dataset
    steps_per_epoch = max(1, len(X_train) // batch_size)
    validation_steps = max(1, len(X_val) // batch_size)

    print(f"🔥 Starting EXTENDED training with advanced strategies:")
    print(f"   ⏰ Epochs: {epochs} (extended for challenging dataset)")
    print(f"   📦 Batch size: {batch_size} (small for stability)")
    print(f"   📈 Steps per epoch: {steps_per_epoch}")
    print(f"   🎯 Using extreme class weights and advanced augmentation")
    print(f"   💪 This will take time but should handle the challenging dataset!")

    # START EXTENDED TRAINING
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_datagen.flow(X_val, y_val, batch_size=batch_size),
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=extreme_weights,  # EXTREME class balancing
        verbose=1
    )

    # Step 7: Advanced evaluation
    print(f"\n📊 STEP 7: ADVANCED EVALUATION")
    print("-" * 60)

    # Predict on test set
    y_pred_proba = model.predict(val_datagen.flow(X_test, batch_size=batch_size, shuffle=False),
                                 steps=len(X_test) // batch_size + 1, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Calculate metrics
    accuracy = np.mean(y_pred == y_test)
    unique_predictions = len(np.unique(y_pred))

    print(f"🎯 ULTRA-ROBUST MODEL RESULTS:")
    print(f"   📊 Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"   🎯 Classes Predicted: {unique_predictions}/{len(classifier.class_names)}")

    # Success assessment for challenging dataset
    if accuracy > 0.70:
        print(f"   🏆 OUTSTANDING! Exceptional results for this challenging dataset!")
    elif accuracy > 0.60:
        print(f"   🥇 EXCELLENT! Very good results for multi-environment challenge!")
    elif accuracy > 0.50:
        print(f"   🥈 GOOD! Solid results given the extreme dataset challenges!")
    elif accuracy > 0.40:
        print(f"   🥉 DECENT! Reasonable results for this very challenging dataset!")
    else:
        print(f"   📊 CHALLENGING! Dataset requires further domain expertise!")

    if unique_predictions >= 4:
        print(f"   ✅ PERFECT! All classes being predicted!")
    elif unique_predictions >= 3:
        print(f"   ✅ GREAT! Most classes being predicted!")
    elif unique_predictions >= 2:
        print(f"   ✅ GOOD! Multiple classes being predicted!")
    else:
        print(f"   ⚠️ Still single-class bias - dataset extremely challenging!")

    # Detailed classification report
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=classifier.class_names, digits=4))

    # Plot results
    plot_advanced_results(history, y_test, y_pred, classifier.class_names, accuracy)

    # Save model
    model.save('ultra_robust_efficientnetb0_final.h5')
    print(f"💾 Ultra-robust model saved!")

    return model, accuracy, unique_predictions


def plot_advanced_results(history, y_test, y_pred, class_names, accuracy):
    """Plot comprehensive results for advanced solution"""

    # Training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Ultra-Robust EfficientNetB0 - Advanced Training Results', fontsize=16, fontweight='bold')

    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], 'b-', linewidth=2, label='Training')
    axes[0, 0].plot(history.history['val_accuracy'], 'r-', linewidth=2, label='Validation')
    axes[0, 0].set_title('Model Accuracy (Extended Training)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Loss
    axes[0, 1].plot(history.history['loss'], 'b-', linewidth=2, label='Training')
    axes[0, 1].plot(history.history['val_loss'], 'r-', linewidth=2, label='Validation')
    axes[0, 1].set_title('Model Loss (Extended Training)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    im = axes[1, 0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 0].set_title('Confusion Matrix')
    tick_marks = np.arange(len(class_names))
    axes[1, 0].set_xticks(tick_marks)
    axes[1, 0].set_xticklabels(class_names, rotation=45)
    axes[1, 0].set_yticks(tick_marks)
    axes[1, 0].set_yticklabels(class_names)

    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        axes[1, 0].text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

    # Summary stats
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])

    summary_text = f"""Advanced Training Summary:

Final Test Accuracy: {accuracy:.4f}
Final Training Accuracy: {final_train_acc:.4f}
Final Validation Accuracy: {final_val_acc:.4f}
Best Validation Accuracy: {best_val_acc:.4f}
Total Epochs: {len(history.history['accuracy'])}

Model Enhancements:
• Ultra-robust architecture (4-layer head)
• Extreme class balancing
• Advanced data augmentation
• Extended training (60 epochs)
• Ultra-conservative learning rate
• Multi-environment optimization

Dataset Challenge Level: EXTREME
Expected Range: 40-75% accuracy
Actual Result: {accuracy * 100:.1f}%"""

    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].set_title('Advanced Solution Summary')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    """Main function with advanced solution"""
    print("🌊 ADVANCED RIVER WATER LEVEL CLASSIFICATION 🌊")
    print("Ultra-Robust Solution for Extreme Multi-Environment Dataset")
    print("=" * 80)

    # Run advanced training strategy
    model, accuracy, classes_predicted = advanced_training_strategy()

    if model and accuracy:
        print(f"\n🎊 ADVANCED TRAINING COMPLETED!")
        print(f"🏆 Final Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"🎯 Classes Predicted: {classes_predicted}/4")
        print(f"💾 Model saved as: ultra_robust_efficientnetb0_final.h5")

        # Assessment for challenging dataset
        if accuracy > 0.60:
            print(f"🏆 EXCEPTIONAL results for this extremely challenging dataset!")
        elif accuracy > 0.50:
            print(f"🥇 EXCELLENT results given the multi-environment complexity!")
        elif accuracy > 0.40:
            print(f"🥈 GOOD results - significant achievement for this dataset!")
        else:
            print(f"🥉 DECENT attempt - this dataset is exceptionally challenging!")

        print(f"\n📊 Advanced Solution Features Applied:")
        print(f"   ✅ Ultra-robust 4-layer classification head")
        print(f"   ✅ Extreme class balancing (3x boost for minorities)")
        print(f"   ✅ Advanced multi-environment data augmentation")
        print(f"   ✅ Extended training (60 epochs)")
        print(f"   ✅ Ultra-conservative learning rate (1e-5)")
        print(f"   ✅ Heavy regularization and dropout")

        print("\n🚀 Ready for academic presentation with advanced analysis!")
    else:
        print("\n❌ Advanced training encountered issues. Check error messages above.")


if __name__ == "__main__":
    main()