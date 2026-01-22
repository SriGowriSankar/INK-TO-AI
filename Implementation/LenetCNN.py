import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

print(f'TensorFlow version: {tf.__version__}')

# ==================== PREPROCESSING FUNCTIONS ====================

def center_image(img, size=(28, 28), bg=255):
    """Center and resize image to target size"""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    im = img.convert('L')
    arr = np.array(im)

    # Find foreground by threshold
    th = arr.mean() * 0.9
    mask = arr < th

    if not mask.any():
        return im.resize(size, Image.BILINEAR)

    # Get bounding box
    ys, xs = np.where(mask)
    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()

    # Crop to content
    crop = im.crop((xmin, ymin, xmax + 1, ymax + 1))
    cw, ch = crop.size
    target_w, target_h = size

    # Scale to fit
    scale = min(target_w / cw, target_h / ch)
    new_size = (max(1, int(cw * scale)), max(1, int(ch * scale)))
    crop = crop.resize(new_size, Image.BILINEAR)

    # Paste on canvas
    canvas = Image.new('L', size, color=bg)
    ox = (target_w - new_size[0]) // 2
    oy = (target_h - new_size[1]) // 2
    canvas.paste(crop, (ox, oy))

    return canvas


def deskew_and_center(img, size=(28, 28)):
    """Deskew image using moments and center it"""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    im = img.convert('L')
    arr = np.array(im).astype(np.float32)

    # Invert for processing
    arr_inv = 255.0 - arr
    thr = arr_inv.mean() + 10
    bw = arr_inv > thr

    if not bw.any():
        return center_image(im, size=size)

    # Calculate moments for deskewing
    ys, xs = np.where(bw)
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    x_c = x - x.mean()
    y_c = y - y.mean()

    mu11 = (x_c * y_c).mean()
    mu20 = (x_c ** 2).mean()
    mu02 = (y_c ** 2).mean()

    denom = (mu20 - mu02)
    angle = 0.0 if denom == 0 else 0.5 * np.arctan2(2 * mu11, denom)
    angle_deg = np.degrees(angle)

    # Rotate to deskew
    rotated = im.rotate(-angle_deg, resample=Image.BILINEAR, fillcolor=255)

    return center_image(rotated, size=size)


def preprocess_image(img_path):
    """Load and preprocess image for prediction"""
    img = Image.open(img_path).convert('L')
    processed = deskew_and_center(img, size=(28, 28))
    arr = np.array(processed).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=-1)
    return arr, processed


# ==================== MODEL ARCHITECTURES ====================

def build_lenet_classic(input_shape=(28, 28, 1), num_classes=62):
    """Classic LeNet architecture"""
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(6, kernel_size=5, activation='tanh', padding='same')(inputs)
    x = layers.AveragePooling2D(pool_size=2)(x)
    x = layers.Conv2D(16, kernel_size=5, activation='tanh')(x)
    x = layers.AveragePooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(120, activation='tanh')(x)
    x = layers.Dense(84, activation='tanh')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='LeNet_Classic')
    return model


def build_lenet_modern(input_shape=(28, 28, 1), num_classes=62):
    """Modern LeNet with BatchNorm and ReLU"""
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='LeNet_Modern')
    return model


def build_lenet_deep(input_shape=(28, 28, 1), num_classes=62):
    """Deeper architecture for better accuracy"""
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='LeNet_Deep')
    return model


def get_model(architecture='modern', input_shape=(28, 28, 1), num_classes=62):
    """Get model by architecture name"""
    if architecture == 'classic':
        return build_lenet_classic(input_shape, num_classes)
    elif architecture == 'modern':
        return build_lenet_modern(input_shape, num_classes)
    elif architecture == 'deep':
        return build_lenet_deep(input_shape, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# ==================== DATA LOADING ====================

def load_emnist_data(split='byclass'):
    """Load EMNIST dataset using Keras"""
    print(f"Loading EMNIST {split}...")

    # For demonstration, we'll use MNIST
    # In production, use tensorflow_datasets for EMNIST
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Reshape and normalize
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # Class names for MNIST
    class_names = [str(i) for i in range(10)]

    print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")
    print(f"Classes: {len(class_names)}")

    return (x_train, y_train), (x_test, y_test), class_names


def create_data_augmentation():
    """Create data augmentation layer"""
    return keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
    ])


# ==================== TRAINING ====================

def train_model(architecture='modern', epochs=20, batch_size=128, save_dir='./models'):
    """Train the model"""
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    (x_train, y_train), (x_test, y_test), class_names = load_emnist_data()
    num_classes = len(class_names)

    # Build model
    model = get_model(architecture, num_classes=num_classes)

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(save_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-7
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\n" + "="*60)
    print("Final evaluation...")
    print("="*60 + "\n")

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Save
    model.save(os.path.join(save_dir, 'final_model.h5'))
    with open(os.path.join(save_dir, 'class_names.json'), 'w') as f:
        json.dump(class_names, f)

    print(f"\nModel saved to {save_dir}")

    # Plot training history
    plot_training_history(history, save_dir)

    return model, history, class_names


def plot_training_history(history, save_dir='./models'):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    print(f"Training history plot saved to {save_dir}/training_history.png")
    plt.show()


# ==================== PREDICTION ====================

class HandwritingRecognizer:
    """Handwriting recognition predictor"""

    def __init__(self, model_path='./models/best_model.h5',
                 class_names_path='./models/class_names.json'):
        self.model = keras.models.load_model(model_path)
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        print(f"Model loaded from {model_path}")
        print(f"Classes: {len(self.class_names)}")

    def predict(self, img_path, top_k=3, show_image=True):
        """Predict character from image"""
        # Preprocess
        img_array, processed_img = preprocess_image(img_path)
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = self.model.predict(img_array, verbose=0)[0]

        # Get top k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        results = [
            (self.class_names[i], float(predictions[i]))
            for i in top_indices
        ]

        # Display
        if show_image:
            self._display_prediction(img_path, processed_img, results)

        return results

    def _display_prediction(self, original_path, processed_img, results):
        """Display prediction results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Original
        orig_img = Image.open(original_path).convert('L')
        ax1.imshow(orig_img, cmap='gray')
        ax1.set_title('Original Image')
        ax1.axis('off')

        # Processed
        ax2.imshow(processed_img, cmap='gray')
        title = f"Processed (28x28)\n"
        title += f"Prediction: {results[0][0]} ({results[0][1]:.2%})"
        ax2.set_title(title)
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

        # Print results
        print("\nTop predictions:")
        for i, (label, conf) in enumerate(results, 1):
            print(f"{i}. {label}: {conf:.2%}")

    def predict_batch(self, image_paths, show_grid=True):
        """Predict multiple images"""
        results = []
        for path in image_paths:
            result = self.predict(path, show_image=False)
            results.append((path, result))

        if show_grid:
            self._display_batch_predictions(results)

        return results

    def _display_batch_predictions(self, results):
        """Display batch prediction results"""
        n = len(results)
        cols = min(4, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        if rows == 1:
            axes = [axes]
        if cols == 1:
            axes = [[ax] for ax in axes]

        for idx, (path, preds) in enumerate(results):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col]

            img = Image.open(path).convert('L')
            ax.imshow(img, cmap='gray')
            title = f"{preds[0][0]} ({preds[0][1]:.1%})"
            ax.set_title(title)
            ax.axis('off')

        # Hide empty subplots
        for idx in range(len(results), rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row][col].axis('off')

        plt.tight_layout()
        plt.show()


# ==================== MAIN INTERFACE ====================

def main():
    """Main execution function"""
    print("="*60)
    print("HANDWRITING RECOGNITION SYSTEM")
    print("="*60)
    print("\nOptions:")
    print("1. Train new model")
    print("2. Load model and predict")
    print("3. Quick demo with sample images")
    print("="*60)

    choice = input("\nEnter your choice (1/2/3): ").strip()

    if choice == '1':
        print("\n--- Training Configuration ---")
        architecture = input("Architecture (classic/modern/deep) [modern]: ").strip() or 'modern'
        epochs = int(input("Number of epochs [20]: ").strip() or '20')
        batch_size = int(input("Batch size [128]: ").strip() or '128')

        model, history, class_names = train_model(
            architecture=architecture,
            epochs=epochs,
            batch_size=batch_size
        )

        print("\n✓ Training complete!")
        print("  Model saved to ./models/")

    elif choice == '2':
        model_path = input("Model path [./models/best_model.h5]: ").strip() or './models/best_model.h5'
        class_names_path = input("Class names path [./models/class_names.json]: ").strip() or './models/class_names.json'

        recognizer = HandwritingRecognizer(model_path, class_names_path)

        while True:
            img_path = input("\nEnter image path (or 'q' to quit): ").strip()
            if img_path.lower() == 'q':
                break

            if os.path.exists(img_path):
                recognizer.predict(img_path)
            else:
                print(f"Error: File not found: {img_path}")

    elif choice == '3':
        print("\n--- Quick Demo ---")
        print("This will train a small model on MNIST for 5 epochs")
        print("and show predictions on test samples.")

        # Quick training
        model, history, class_names = train_model(
            architecture='modern',
            epochs=5,
            batch_size=128
        )

        # Load test samples
        print("\nLoading test samples...")
        (_, _), (x_test, y_test), _ = load_emnist_data()

        # Show random predictions
        indices = np.random.choice(len(x_test), 9, replace=False)
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))

        for idx, ax in zip(indices, axes.flat):
            img = x_test[idx]
            pred = model.predict(img[np.newaxis, ...], verbose=0)
            pred_class = class_names[np.argmax(pred)]
            true_class = class_names[y_test[idx]]

            ax.imshow(img.squeeze(), cmap='gray')
            color = 'green' if pred_class == true_class else 'red'
            ax.set_title(f"True: {true_class}\nPred: {pred_class}", color=color)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('./models/demo_predictions.png', dpi=150)
        plt.show()

        print("\n✓ Demo complete!")
        print("  Results saved to ./models/demo_predictions.png")

    else:
        print("Invalid choice!")


if __name__ == '__main__':
    # Auto-run Option 3: Quick Demo
    print("="*60)
    print("HANDWRITING RECOGNITION SYSTEM - QUICK DEMO")
    print("="*60)
    print("\nThis will:")
    print("✓ Train a modern CNN on MNIST for 5 epochs")
    print("✓ Show training progress and accuracy")
    print("✓ Display predictions on 9 random test samples")
    print("✓ Save model and results to ./models/")
    print("\n" + "="*60 + "\n")

    # Quick training
    print("STEP 1: Training model...")
    model, history, class_names = train_model(
        architecture='modern',
        epochs=5,
        batch_size=128
    )

    # Load test samples
    print("\n" + "="*60)
    print("STEP 2: Testing on sample images...")
    print("="*60 + "\n")

    (_, _), (x_test, y_test), _ = load_emnist_data()

    # Show random predictions
    indices = np.random.choice(len(x_test), 9, replace=False)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    print("Generating predictions for 9 random samples...\n")

    correct = 0
    for idx, ax in zip(indices, axes.flat):
        img = x_test[idx]
        pred = model.predict(img[np.newaxis, ...], verbose=0)
        pred_probs = pred[0]
        pred_class_idx = np.argmax(pred_probs)
        pred_class = class_names[pred_class_idx]
        true_class = class_names[y_test[idx]]
        confidence = pred_probs[pred_class_idx]

        # Count correct predictions
        if pred_class == true_class:
            correct += 1

        ax.imshow(img.squeeze(), cmap='gray')
        color = 'green' if pred_class == true_class else 'red'
        ax.set_title(
            f"True: {true_class} | Pred: {pred_class}\nConfidence: {confidence:.2%}",
            color=color,
            fontsize=10,
            fontweight='bold'
        )
        ax.axis('off')

        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)

    plt.suptitle(
        f'Random Predictions - Accuracy: {correct}/9 ({correct/9:.1%})',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    plt.tight_layout()
    plt.savefig('./models/demo_predictions.png', dpi=150, bbox_inches='tight')
    print(f"✓ Predictions saved to ./models/demo_predictions.png")
    plt.show()

    # Summary
    print("\n" + "="*60)
    print("DEMO SUMMARY")
    print("="*60)
    print(f"✓ Model Architecture: Modern CNN")
    print(f"✓ Training Epochs: 5")
    print(f"✓ Final Training Accuracy: {history.history['accuracy'][-1]:.2%}")
    print(f"✓ Final Validation Accuracy: {history.history['val_accuracy'][-1]:.2%}")
    print(f"✓ Sample Predictions Correct: {correct}/9 ({correct/9:.1%})")
    print(f"\n✓ All files saved to ./models/")
    print(f"  - best_model.h5 (best model during training)")
    print(f"  - final_model.h5 (final model)")
    print(f"  - class_names.json (class labels)")
    print(f"  - training_history.png (accuracy/loss plots)")
    print(f"  - demo_predictions.png (sample predictions)")
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)

    # Show how to use the model
    print("\n TO USE THIS MODEL FOR PREDICTIONS:")
    print("-" * 60)
    print("from PIL import Image")
    print("recognizer = HandwritingRecognizer(")
    print("    model_path='./models/best_model.h5',")
    print("    class_names_path='./models/class_names.json'")
    print(")")
    print("recognizer.predict('your_image.png')")
    print("-" * 60)