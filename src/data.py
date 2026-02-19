
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(data_dir, image_size=(128, 128), batch_size=32, validation_split=0.2):
    """
    Creates and returns data generators for training and validation.

    Args:
        data_dir (str): Path to the directory containing raw image data.
        image_size (tuple): Target size for resizing images (height, width).
        batch_size (int): Number of images per batch.
        validation_split (float): Fraction of images to reserve for validation.

    Returns:
        tuple: A tuple containing (train_generator, validation_generator).
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print(f"Setting up data generators for directory: {data_dir}")
    print(f"Image size: {image_size}, Batch size: {batch_size}, Validation split: {validation_split}")

    # Data Augmentation and Preprocessing for training data
    train_datagen = ImageDataGenerator(
        rescale=1./255, # Normalize pixel values to [0, 1]
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split # Specify validation split
    )

    # Preprocessing for validation data (only rescaling)
    val_datagen = ImageDataGenerator(
        rescale=1./255, # Normalize pixel values to [0, 1]
        validation_split=validation_split
    )

    # Flow training images in batches from directory
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        seed=42
    )

    # Flow validation images in batches from directory
    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        seed=42
    )

    print("Data generators created successfully.")
    return train_generator, validation_generator

if __name__ == '__main__':
    # Example usage: create dummy data for testing
    print("Creating dummy data for testing data.py...")
    dummy_data_root = 'dummy_data_for_test'
    dummy_train_dir = os.path.join(dummy_data_root, 'raw')

    # Create dummy classes and images
    os.makedirs(os.path.join(dummy_train_dir, 'class_0'), exist_ok=True)
    os.makedirs(os.path.join(dummy_train_dir, 'class_1'), exist_ok=True)
    tf.random.set_seed(42)
    for i in range(20):
        # Create dummy image files with tf.float32, then cast to tf.uint8 for PNG encoding
        dummy_image = tf.random.uniform(shape=[50, 50, 3], minval=0., maxval=255., dtype=tf.float32)
        tf.io.write_file(os.path.join(dummy_train_dir, 'class_0', f'image_{i}.png'), tf.io.encode_png(tf.cast(dummy_image, tf.uint8)))
        dummy_image = tf.random.uniform(shape=[50, 50, 3], minval=0., maxval=255., dtype=tf.float32)
        tf.io.write_file(os.path.join(dummy_train_dir, 'class_1', f'image_{i}.png'), tf.io.encode_png(tf.cast(dummy_image, tf.uint8)))
    print("Dummy data created.")

    # Test the data generators
    try:
        train_gen, val_gen = get_data_generators(dummy_train_dir, image_size=(50, 50))
        print(f"Found {train_gen.samples} training images belonging to {train_gen.num_classes} classes.")
        print(f"Found {val_gen.samples} validation images belonging to {val_gen.num_classes} classes.")
        print(f"Training batches: {len(train_gen)}, Validation batches: {len(val_gen)}")
        # Clean up dummy data
        import shutil
        shutil.rmtree(dummy_data_root)
        print("Dummy data cleaned up.")
    except Exception as e:
        print(f"Error testing data generators: {e}")

