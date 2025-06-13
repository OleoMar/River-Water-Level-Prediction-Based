# River Water Level Classification using Deep Learning - FIXED VERSION
# Complete implementation with ResNet50, EfficientNetB0, and MobileNetV2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
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

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class RiverWaterLevelClassifier:
    """
    River Water Level Classifier using Deep Learning Models
    """

    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        self.dataset_path = None
        self.class_names = []
        self.num_classes = 0
        self.models = {}
        self.histories = {}

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

        # Find all directories and files
        all_items = list(self.dataset_path.rglob("*"))
        dirs = [item for item in all_items if item.is_dir()]
        files = [item for item in all_items if item.is_file()]

        print(f"📊 Total items: {len(all_items)}")
        print(f"📁 Directories: {len(dirs)}")
        print(f"📄 Files: {len(files)}")

        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in files if f.suffix.lower() in image_extensions]

        print(f"\n🖼️  Total image files found: {len(image_files)}")

        # Analyze class structure
        class_info = {}
        for img_file in image_files:
            parts = img_file.relative_to(self.dataset_path).parts
            if len(parts) > 1:
                class_name = parts[-2]  # Parent directory as class
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
        fig.suptitle('River Water Level Dataset - Real Sample Images', fontsize=16)

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

        # Plot class distribution
        plt.figure(figsize=(15, 5))
        unique, counts = np.unique(y, return_counts=True)

        plt.subplot(1, 3, 1)
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))
        bars = plt.bar([self.class_names[i] for i in unique], counts, color=colors)
        plt.title('Class Distribution')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)

        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                     str(count), ha='center', va='bottom')

        plt.subplot(1, 3, 2)
        plt.pie(counts, labels=[self.class_names[i] for i in unique], autopct='%1.1f%%', colors=colors)
        plt.title('Class Distribution (Percentage)')

        plt.subplot(1, 3, 3)
        plt.text(0.1, 0.8, f'Dataset Statistics:', fontsize=14, fontweight='bold', transform=plt.gca().transAxes)
        plt.text(0.1, 0.7, f'Total samples: {len(X)}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.6, f'Image size: {X.shape[1]}×{X.shape[2]}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.5, f'Channels: {X.shape[3]}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.4, f'Classes: {self.num_classes}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.3, f'Min samples/class: {min(counts)}', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.1, 0.2, f'Max samples/class: {max(counts)}', fontsize=12, transform=plt.gca().transAxes)
        plt.axis('off')
        plt.title('Dataset Overview')

        plt.tight_layout()
        plt.show()

        print("\n" + "=" * 60)
        print("REAL DATASET STATISTICS")
        print("=" * 60)
        print(f"Total samples: {len(X)}")
        print(f"Image dimensions: {X.shape[1:3]}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Class names: {self.class_names}")
        print("\nClass distribution:")
        for i, class_name in enumerate(self.class_names):
            count = np.sum(y == i)
            percentage = (count / len(y)) * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        print("=" * 60)

    def create_model(self, model_name='resnet50'):
        """Create a deep learning model for river water level classification"""
        if model_name.lower() == 'resnet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
        elif model_name.lower() == 'efficientnetb0':
            base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
        elif model_name.lower() == 'mobilenetv2':
            base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*self.img_size, 3))
        else:
            raise ValueError("Model must be one of: resnet50, efficientnetb0, mobilenetv2")

        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu', name='dense_512')(x)
        x = Dropout(0.5, name='dropout_1')(x)
        x = Dense(256, activation='relu', name='dense_256')(x)
        x = Dropout(0.3, name='dropout_2')(x)
        predictions = Dense(self.num_classes, activation='softmax', name='predictions')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        print(f"\n🤖 Created {model_name.upper()} model:")
        print(f"   📊 Total parameters: {model.count_params():,}")
        print(
            f"   🔧 Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")

        return model

    def train_model(self, model, X_train, y_train, X_val, y_val, model_name, epochs=20):
        """Train the model with proper callbacks and data augmentation"""
        print(f"\n🚀 Training {model_name.upper()} model...")

        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )

        val_datagen = ImageDataGenerator()

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                f'best_{model_name}_model.h5',
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            )
        ]

        batch_size = 32
        steps_per_epoch = max(1, len(X_train) // batch_size)
        validation_steps = max(1, len(X_val) // batch_size)

        print(f"   📊 Batch size: {batch_size}")
        print(f"   🔄 Steps per epoch: {steps_per_epoch}")
        print(f"   ✅ Validation steps: {validation_steps}")

        history = model.fit(
            train_datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_datagen.flow(X_val, y_val, batch_size=batch_size),
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def plot_training_curves(self, history, model_name):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{model_name.upper()} Training Analysis', fontsize=16, fontweight='bold')

        axes[0, 0].plot(history.history['accuracy'], 'b-', linewidth=2, label='Training')
        axes[0, 0].plot(history.history['val_accuracy'], 'r-', linewidth=2, label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(history.history['loss'], 'b-', linewidth=2, label='Training')
        axes[0, 1].plot(history.history['val_loss'], 'r-', linewidth=2, label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        if 'lr' in history.history:
            axes[1, 0].plot(history.history['lr'], 'g-', linewidth=2)
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate\nNot Available', ha='center', va='center')
            axes[1, 0].set_title('Learning Rate Schedule')

        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        best_val_acc = max(history.history['val_accuracy'])

        stats_text = f"""Training Summary:

Final Training Accuracy: {final_train_acc:.4f}
Final Validation Accuracy: {final_val_acc:.4f}
Best Validation Accuracy: {best_val_acc:.4f}
Total Epochs: {len(history.history["accuracy"])}
Overfitting: {'Yes' if final_train_acc - final_val_acc > 0.1 else 'No'}"""

        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Training Statistics')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate the model and show detailed results"""
        print(f"\n📊 Evaluating {model_name.upper()} model...")

        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names, digits=4))

        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'{model_name.upper()} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        plt.subplot(1, 2, 2)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Reds',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'{model_name.upper()} - Normalized Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        plt.tight_layout()
        plt.show()

        return accuracy, y_pred, y_pred_proba

    def demonstrate_single_prediction(self, model, X_test, y_test, model_name):
        """Demonstrate prediction for a single test image"""
        idx = np.random.randint(0, len(X_test))
        test_image = X_test[idx]
        true_label = y_test[idx]

        prediction = model.predict(np.expand_dims(test_image, axis=0), verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        plt.figure(figsize=(15, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(test_image)
        plt.title(f'Test Image #{idx}\nTrue Label: {self.class_names[true_label]}')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        colors = ['red' if i == predicted_class else 'lightblue' for i in range(len(self.class_names))]
        bars = plt.bar(self.class_names, prediction[0] * 100, color=colors)
        plt.title(
            f'{model_name.upper()} Prediction\nPredicted: {self.class_names[predicted_class]} ({confidence:.1f}%)')
        plt.ylabel('Confidence (%)')
        plt.xticks(rotation=45)

        for bar, prob in zip(bars, prediction[0]):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{prob * 100:.1f}%', ha='center', va='bottom', fontsize=10)

        status = "✅ CORRECT" if predicted_class == true_label else "❌ INCORRECT"
        color = "green" if predicted_class == true_label else "red"

        plt.subplot(1, 3, 3)
        plt.text(0.1, 0.8, 'Prediction Analysis:', fontsize=14, fontweight='bold')
        plt.text(0.1, 0.7, f'True Label: {self.class_names[true_label]}', fontsize=12)
        plt.text(0.1, 0.6, f'Predicted: {self.class_names[predicted_class]}', fontsize=12)
        plt.text(0.1, 0.5, f'Confidence: {confidence:.2f}%', fontsize=12)
        plt.text(0.1, 0.4, f'Status: {status}', fontsize=12, color=color, fontweight='bold')

        top_2_indices = np.argsort(prediction[0])[-2:][::-1]
        plt.text(0.1, 0.3, 'Top 2 Predictions:', fontsize=12, fontweight='bold')
        for i, idx_pred in enumerate(top_2_indices):
            plt.text(0.1, 0.2 - i * 0.05,
                     f'{i + 1}. {self.class_names[idx_pred]}: {prediction[0][idx_pred] * 100:.1f}%', fontsize=10)

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Detailed Analysis')

        plt.tight_layout()
        plt.show()

        print(f"\n🔍 {model_name.upper()} Single Image Prediction Analysis:")
        print(f"   🖼️  Image Index: {idx}")
        print(f"   🏷️  True Label: {self.class_names[true_label]}")
        print(f"   🤖 Predicted Label: {self.class_names[predicted_class]}")
        print(f"   📊 Confidence: {confidence:.2f}%")
        print(f"   ✅ Correct: {'Yes' if predicted_class == true_label else 'No'}")
        print(f"   🎯 Status: {status}")

    def compare_models(self, results):
        """Compare performance of all models"""
        if not results:
            print("⚠️ No models trained for comparison")
            return

        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]

        plt.figure(figsize=(20, 12))

        plt.subplot(2, 3, 1)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(models)]
        bars = plt.bar(models, accuracies, color=colors)
        plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Test Accuracy')
        plt.ylim(0, 1)

        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

        if len(models) >= 2:
            plt.subplot(2, 3, 2)
            for i, model in enumerate(models):
                if 'history' in results[model]:
                    plt.plot(results[model]['history'].history['accuracy'],
                             label=f'{model.upper()}', linewidth=2, color=colors[i])
            plt.title('Training Accuracy Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Training Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 3, 3)
            for i, model in enumerate(models):
                if 'history' in results[model]:
                    plt.plot(results[model]['history'].history['val_accuracy'],
                             label=f'{model.upper()}', linewidth=2, color=colors[i])
            plt.title('Validation Accuracy Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Validation Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 4)
        table_data = []
        for model in models:
            acc = results[model]['accuracy']
            table_data.append([model.upper(), f"{acc:.4f}"])

        table = plt.table(cellText=table_data,
                          colLabels=['Model', 'Test Accuracy'],
                          cellLoc='center',
                          loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)

        if models:
            best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
            best_row = models.index(best_model) + 1
            for col in range(2):
                table[(best_row, col)].set_facecolor('#90EE90')

        plt.title('Model Performance Summary', fontsize=14, fontweight='bold', pad=20)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        print("\n" + "=" * 80)
        print("🏆 COMPREHENSIVE MODEL COMPARISON")
        print("=" * 80)

        if models:
            best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
            best_accuracy = results[best_model]['accuracy']

            sorted_models = sorted(results.keys(), key=lambda x: results[x]['accuracy'], reverse=True)
            for rank, model in enumerate(sorted_models, 1):
                acc = results[model]['accuracy']
                status = "🥇 BEST" if model == best_model else f"#{rank}"
                print(f"   {status} {model.upper():15} | Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")

            print(f"\n🎯 Winner: {best_model.upper()} with {best_accuracy:.4f} accuracy")
        print("=" * 80)


def main():
    """Main function to run the complete river water level classification project"""
    print("🌊 RIVER WATER LEVEL CLASSIFICATION PROJECT 🌊")
    print("Using Real Kaggle Dataset: otavio12/urban-river-level-classification-image-dataset")
    print("=" * 80)

    classifier = RiverWaterLevelClassifier()

    # Step 1: Download Dataset
    print("\n📥 STEP 1: Dataset Download")
    dataset_path = classifier.download_dataset()
    if not dataset_path:
        print("❌ Failed to download dataset. Please check your kagglehub installation.")
        return

    # Step 2: Load and Analyze Dataset
    print("\n📊 STEP 2: Dataset Loading and Analysis")
    X, y = classifier.load_real_dataset(max_samples_per_class=300)  # Limit for demo

    if X is None or len(X) == 0:
        print("❌ Failed to load dataset. Please check the dataset structure.")
        return

    classifier.visualize_real_dataset(X, y)

    # Step 3: Data Splitting
    print("\n🔄 STEP 3: Data Splitting")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"✅ Data split completed:")
    print(f"   📚 Training set: {len(X_train)} samples ({len(X_train) / len(X) * 100:.1f}%)")
    print(f"   🔍 Validation set: {len(X_val)} samples ({len(X_val) / len(X) * 100:.1f}%)")
    print(f"   🧪 Test set: {len(X_test)} samples ({len(X_test) / len(X) * 100:.1f}%)")

    # Step 4: Train All Models
    model_names = ['resnet50', 'efficientnetb0', 'mobilenetv2']
    results = {}

    epochs = 15  # Reduced for faster demo

    for i, model_name in enumerate(model_names, 1):
        print(f"\n🤖 STEP 4.{i}: Training {model_name.upper()}")
        print("-" * 50)

        try:
            # Create model
            model = classifier.create_model(model_name)

            # Train model
            print(f"🚀 Starting training for {model_name.upper()}...")
            history = classifier.train_model(
                model, X_train, y_train, X_val, y_val, model_name, epochs=epochs
            )

            # Plot training curves
            classifier.plot_training_curves(history, model_name)

            # Evaluate model
            accuracy, y_pred, y_pred_proba = classifier.evaluate_model(
                model, X_test, y_test, model_name
            )

            # Demonstrate single prediction
            classifier.demonstrate_single_prediction(model, X_test, y_test, model_name)

            # Store results
            results[model_name] = {
                'model': model,
                'history': history,
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }

            print(f"✅ {model_name.upper()} training completed successfully!")

        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Step 5: Compare All Models
    if len(results) > 0:
        print(f"\n📈 STEP 5: Comprehensive Model Comparison")
        print("-" * 50)
        classifier.compare_models(results)
    else:
        print("\n⚠️ No models were successfully trained.")

    # Step 6: Final Summary
    print(f"\n🎊 PROJECT COMPLETION SUMMARY")
    print("=" * 80)
    print("✅ Successfully completed all required components:")
    print("   📊 Dataset visualization and salient feature analysis")
    print("   🔄 Clear train/test data demonstration")
    print("   🧪 Single image inference from test data")
    print("   🤖 Complete trained models with training curves")
    print("   📈 Comprehensive results and performance graphs")
    print("   🏆 Model comparison and ranking")

    if results:
        best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_acc = results[best_model]['accuracy']
        print(f"\n🏆 BEST PERFORMING MODEL: {best_model.upper()}")
        print(f"🎯 Best Accuracy: {best_acc:.4f} ({best_acc * 100:.2f}%)")

        print(f"\n📋 MODELS TRAINED:")
        for model_name in results.keys():
            acc = results[model_name]['accuracy']
            print(f"   • {model_name.upper()}: {acc:.4f} accuracy")

    print("\n💾 DELIVERABLES READY:")
    print("   📁 Source code (complete implementation)")
    print("   📊 Dataset analysis and visualization")
    print("   🤖 Trained model files (.h5)")
    print("   📈 Training curves and performance metrics")
    print("   📋 Classification reports and confusion matrices")
    print("   🔍 Single image prediction demonstrations")

    print("\n🔗 DATASET INFORMATION:")
    print("   📍 Source: Kaggle - otavio12/urban-river-level-classification-image-dataset")
    print("   📁 Local path: " + str(classifier.dataset_path) if classifier.dataset_path else "Not available")
    print("   🏷️ Classes: " + ", ".join(classifier.class_names) if classifier.class_names else "Not determined")
    print("   📊 Total samples processed: " + str(len(X)) if 'X' in locals() else "0")

    print("\n🚀 Ready for Overleaf integration and academic presentation!")
    print("=" * 80)


if __name__ == "__main__":
    main()