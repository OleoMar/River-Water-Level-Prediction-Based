# ResNet50 Model for River Water Level Classification
# Individual implementation file for Overleaf repository

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class ResNet50Classifier:
    """
    ResNet50-based classifier for river water level prediction

    Architecture Features:
    - Pre-trained ResNet50 backbone (ImageNet weights)
    - Custom classification head with dropout regularization
    - Transfer learning with progressive unfreezing
    - Advanced data augmentation
    """

    def __init__(self, num_classes, img_size=(224, 224)):
        self.num_classes = num_classes
        self.img_size = img_size
        self.model = None
        self.history = None

    def build_model(self):
        """
        Build ResNet50 model with custom classification head

        Architecture:
        - ResNet50 base (frozen initially)
        - GlobalAveragePooling2D
        - Dense(512) + ReLU + Dropout(0.5)
        - Dense(256) + ReLU + Dropout(0.3)
        - Dense(num_classes) + Softmax
        """
        print("🔧 Building ResNet50 model...")

        # Load pre-trained ResNet50
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )

        # Freeze base model initially
        base_model.trainable = False

        # Add custom classification head
        inputs = base_model.input
        x = base_model.output

        # Global average pooling
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)

        # First dense layer
        x = Dense(512, activation='relu', name='dense_512')(x)
        x = Dropout(0.5, name='dropout_1')(x)

        # Second dense layer
        x = Dense(256, activation='relu', name='dense_256')(x)
        x = Dropout(0.3, name='dropout_2')(x)

        # Output layer
        outputs = Dense(self.num_classes, activation='softmax', name='predictions')(x)

        # Create model
        self.model = Model(inputs, outputs, name='ResNet50_WaterLevel')

        # Model summary
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])

        print(f"✅ ResNet50 model built successfully!")
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   🔧 Trainable parameters: {trainable_params:,}")
        print(f"   🔒 Frozen parameters: {total_params - trainable_params:,}")

        return self.model

    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer and loss function
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
        Setup data augmentation specifically tuned for water level images
        """
        # Training augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=20,  # Slight rotation for different viewing angles
            width_shift_range=0.2,  # Horizontal shifts
            height_shift_range=0.2,  # Vertical shifts
            horizontal_flip=True,  # Mirror water scenes
            zoom_range=0.2,  # Zoom for different distances
            shear_range=0.1,  # Slight shear transformation
            brightness_range=[0.8, 1.2],  # Lighting variations
            channel_shift_range=20,  # Color variations
            fill_mode='nearest'
        )

        # Validation augmentation (no augmentation)
        val_datagen = ImageDataGenerator()

        return train_datagen, val_datagen

    def setup_callbacks(self, model_save_path='resnet50_best.h5'):
        """
        Setup training callbacks for optimal training
        """
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),

            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=8,
                min_lr=1e-7,
                verbose=1,
                cooldown=5
            ),

            # Save best model
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]

        return callbacks

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the ResNet50 model with data augmentation and callbacks
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print(f"🚀 Starting ResNet50 training...")
        print(f"   📊 Training samples: {len(X_train)}")
        print(f"   🔍 Validation samples: {len(X_val)}")
        print(f"   📦 Batch size: {batch_size}")
        print(f"   🔄 Max epochs: {epochs}")

        # Setup data augmentation
        train_datagen, val_datagen = self.setup_data_augmentation()

        # Setup callbacks
        callbacks = self.setup_callbacks()

        # Calculate steps
        steps_per_epoch = max(1, len(X_train) // batch_size)
        validation_steps = max(1, len(X_val) // batch_size)

        # Train the model
        self.history = self.model.fit(
            train_datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_datagen.flow(X_val, y_val, batch_size=batch_size),
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )

        print(f"✅ ResNet50 training completed!")
        return self.history

    def fine_tune(self, X_train, y_train, X_val, y_val, epochs=20, learning_rate=1e-5):
        """
        Fine-tune the model by unfreezing top layers of ResNet50
        """
        if self.model is None or self.history is None:
            raise ValueError("Model must be trained before fine-tuning")

        print(f"🔧 Starting ResNet50 fine-tuning...")

        # Unfreeze the top layers of ResNet50
        base_model = self.model.layers[1]  # ResNet50 base
        base_model.trainable = True

        # Fine-tune from this layer onwards
        fine_tune_at = len(base_model.layers) - 20

        # Freeze all layers before fine_tune_at
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        print(f"   🔧 Trainable parameters after unfreezing: {trainable_params:,}")

        # Setup callbacks for fine-tuning
        callbacks = self.setup_callbacks('resnet50_finetuned.h5')

        # Continue training
        history_fine = self.model.fit(
            X_train, y_train,
            batch_size=16,  # Smaller batch size for fine-tuning
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        # Combine histories
        for key in self.history.history:
            self.history.history[key].extend(history_fine.history[key])

        print(f"✅ ResNet50 fine-tuning completed!")
        return history_fine

    def evaluate(self, X_test, y_test, class_names):
        """
        Evaluate the trained ResNet50 model
        """
        if self.model is None:
            raise ValueError("Model not built or trained")

        print(f"📊 Evaluating ResNet50 model...")

        # Make predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)

        print(f"🎯 ResNet50 Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

        # Detailed classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

        return accuracy, y_pred, y_pred_proba

    def plot_training_history(self):
        """
        Plot training curves specific to ResNet50
        """
        if self.history is None:
            print("❌ No training history available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ResNet50 Training Analysis', fontsize=16, fontweight='bold')

        # Training & Validation Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], 'b-', linewidth=2, label='Training')
        axes[0, 0].plot(self.history.history['val_accuracy'], 'r-', linewidth=2, label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Training & Validation Loss
        axes[0, 1].plot(self.history.history['loss'], 'b-', linewidth=2, label='Training')
        axes[0, 1].plot(self.history.history['val_loss'], 'r-', linewidth=2, label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Learning Rate (if available)
        if 'lr' in self.history.history:
            axes[1, 0].plot(self.history.history['lr'], 'g-', linewidth=2)
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate\nNot Available',
                            ha='center', va='center', fontsize=12)
            axes[1, 0].set_title('Learning Rate Schedule')

        # Training Statistics
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        best_val_acc = max(self.history.history['val_accuracy'])

        stats_text = f"""ResNet50 Training Summary:

Final Training Accuracy: {final_train_acc:.4f}
Final Validation Accuracy: {final_val_acc:.4f}
Best Validation Accuracy: {best_val_acc:.4f}
Total Epochs: {len(self.history.history['accuracy'])}
Overfitting: {'Yes' if final_train_acc - final_val_acc > 0.1 else 'No'}

Architecture Highlights:
• Pre-trained on ImageNet
• 50-layer deep residual network
• Skip connections for gradient flow
• Custom classification head
• Dropout regularization"""

        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('ResNet50 Summary')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, y_test, y_pred, class_names):
        """
        Plot confusion matrix for ResNet50 predictions
        """
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(12, 5))

        # Raw confusion matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('ResNet50 - Confusion Matrix (Raw Counts)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Normalized confusion matrix
        plt.subplot(1, 2, 2)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Reds',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('ResNet50 - Confusion Matrix (Normalized)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        plt.tight_layout()
        plt.show()

    def predict_single_image(self, image, class_names, show_plot=True):
        """
        Predict and visualize a single image
        """
        if self.model is None:
            raise ValueError("Model not built or trained")

        # Ensure image is in correct format
        if len(image.shape) == 3:
            image_batch = np.expand_dims(image, axis=0)
        else:
            image_batch = image

        # Make prediction
        prediction = self.model.predict(image_batch, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0]) * 100

        if show_plot:
            plt.figure(figsize=(12, 6))

            # Show image
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title(f'Input Image\nPredicted: {class_names[predicted_class]}')
            plt.axis('off')

            # Show prediction probabilities
            plt.subplot(1, 2, 2)
            colors = ['red' if i == predicted_class else 'lightblue'
                      for i in range(len(class_names))]
            bars = plt.bar(class_names, prediction[0] * 100, color=colors)
            plt.title(f'ResNet50 Prediction Confidence\n{class_names[predicted_class]}: {confidence:.1f}%')
            plt.ylabel('Confidence (%)')
            plt.xticks(rotation=45)

            # Add percentage labels
            for bar, prob in zip(bars, prediction[0]):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f'{prob * 100:.1f}%', ha='center', va='bottom')

            plt.tight_layout()
            plt.show()

        return predicted_class, confidence, prediction[0]

    def save_model(self, filepath='resnet50_water_level.h5'):
        """
        Save the trained model
        """
        if self.model is None:
            raise ValueError("No model to save")

        self.model.save(filepath)
        print(f"💾 ResNet50 model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load a pre-trained model
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"📁 ResNet50 model loaded from {filepath}")
        return self.model


# Example usage function
def train_resnet50_example(X_train, y_train, X_val, y_val, X_test, y_test, class_names):
    """
    Example function showing how to use ResNet50Classifier
    """
    print("🎯 ResNet50 Training Example")
    print("=" * 50)

    # Initialize classifier
    resnet_classifier = ResNet50Classifier(
        num_classes=len(class_names),
        img_size=(224, 224)
    )

    # Build and compile model
    model = resnet_classifier.build_model()
    resnet_classifier.compile_model(learning_rate=0.001)

    # Train model
    history = resnet_classifier.train(
        X_train, y_train, X_val, y_val,
        epochs=30, batch_size=32
    )

    # Plot training curves
    resnet_classifier.plot_training_history()

    # Evaluate model
    accuracy, y_pred, y_pred_proba = resnet_classifier.evaluate(X_test, y_test, class_names)

    # Plot confusion matrix
    resnet_classifier.plot_confusion_matrix(y_test, y_pred, class_names)

    # Save model
    resnet_classifier.save_model('resnet50_final.h5')

    print(f"✅ ResNet50 training example completed!")
    print(f"🎯 Final accuracy: {accuracy:.4f}")

    return resnet_classifier


if __name__ == "__main__":
    print("🤖 ResNet50 Model for River Water Level Classification")
    print("This is a standalone implementation file.")
    print("Import this module and use ResNet50Classifier class for training.")
