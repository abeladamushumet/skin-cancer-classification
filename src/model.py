
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape=(128, 128, 3), num_classes=2):
    """
    Builds a simple Convolutional Neural Network (CNN) model for image classification.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).
        num_classes (int): The number of output classes for classification.

    Returns:
        tf.keras.Model: The compiled Keras model.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax' if num_classes > 1 else 'sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    # Example usage:
    print("Building a sample model for 2 classes with input shape (128, 128, 3)...")
    sample_model = build_model()
    sample_model.summary()
