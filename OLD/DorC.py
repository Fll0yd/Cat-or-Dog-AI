import argparse
import os
import numpy as np
import tensorflow as tf
import logging
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Constants
TRAIN_DIR = 'cats_dogs_dataset/train'
VAL_DIR = 'cats_dogs_dataset/val'
TEST_DIR = 'cats_dogs_dataset/test'
NEW_IMG = 'new_image.jpg'


def initialize_datagen() -> tuple:

    # Initialize image data generators for training, validation, and testing.
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(
        rescale=1.0/255
    )
    test_datagen = ImageDataGenerator(
        rescale=1.0/255
    )
    return train_datagen, val_datagen, test_datagen


def create_model(img_size: tuple) -> Sequential:
    
    # Create a Convolutional Neural Network model.
    model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(*img_size, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])
    return model


def predict_single_image(model: Sequential, img_size: tuple) -> None:
    
    # Predict the class of a single image.
    if not os.path.exists('new_image.jpg'):
        print("'new_image.jpg' does not exist.")
        return
    img = image.load_img('new_image.jpg', target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    prediction = model.predict(np.expand_dims(img_array, axis=0))[0][0]
    print('Cat' if prediction < 0.5 else 'Dog')


def main():
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a simple CNN for image classification.')
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--img_size", type=int, default=128, help="Image dimensions (img_size x img_size)")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--show_summary", action='store_true', help="Show model summary")
    args = parser.parse_args()

    # Constants
    IMG_SIZE = (args.img_size, args.img_size)
    BATCH_SIZE = args.batch_size
    PATIENCE = args.patience

    # Initialize data generators
    train_datagen, val_datagen, test_datagen = initialize_datagen()

    # Check for directory existence
    for dir_path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if not os.path.exists(dir_path):
            logging.warning(f"{dir_path} directory does not exist. Please make sure your dataset directories are correctly set.")
            return

    # Create data generators
    train_data = train_datagen.flow_from_directory('cats_dogs_dataset/train', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
    val_data = val_datagen.flow_from_directory('cats_dogs_dataset/val', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')
    test_data = test_datagen.flow_from_directory('cats_dogs_dataset/test', target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary')


    # Create the model
    model = create_model(IMG_SIZE)

    # Add Callbacks and TensorBoard
    callbacks = [TensorBoard(log_dir="logs"), EarlyStopping(monitor='val_loss', patience=PATIENCE)]

    # Compile and Train the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])
    model.summary()

    if args.show_summary:
        model.summary()
        
    history = model.fit(
    train_data,
    steps_per_epoch=train_data.n // BATCH_SIZE,
    epochs=50,
    validation_data=val_data,
    validation_steps=val_data.n // BATCH_SIZE,
    callbacks=callbacks)

    # Evaluate the model
    metrics = model.evaluate(test_data, steps=test_data.n // BATCH_SIZE)
    test_loss, test_acc, test_prec, test_recall = metrics
    logging.info(f'Test loss: {test_loss}, Test accuracy: {test_acc}, Test precision: {test_prec}, Test recall: {test_recall}')

    # Save metrics to a JSON file
    with open('metrics.json', 'w') as f:
        json.dump({
            'loss': test_loss,
            'accuracy': test_acc,
            'precision': test_prec,
            'recall': test_recall
        }, f)

    # Save the model
    model.save("trained_model.h5")

    # Single image prediction if image exists
    if os.path.exists(NEW_IMG):
        predict_single_image(model, IMG_SIZE)
    else:
        logging.warning(f"{NEW_IMG} does not exist.")

if __name__ == "__main__":
    main()
