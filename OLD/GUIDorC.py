import os
import numpy as np
import tensorflow as tf
import logging
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tkinter import Tk, Label, Button, filedialog

logging.basicConfig(level=logging.INFO)  # Configured Logging


class GUIApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        root.title("CNN Image Classifier")

        self.label = Label(root, text="Welcome to Image Classifier!")
        self.label.pack()

        self.train_button = Button(root, text="Train Model", command=self.train_model)
        self.train_button.pack()

        self.predict_button = Button(root, text="Predict Image", command=self.predict_image)
        self.predict_button.pack()

    def train_model(self):
        logging.info("Training initiated from GUI.")
        main(True)

    def predict_image(self):
        filename = filedialog.askopenfilename()
        if filename:
            predict_single_image(self.model, IMG_SIZE, filename)
        else:
            logging.warning("No file selected.")


def initialize_datagen() -> tuple:
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
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(*img_size, 3)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

def predict_single_image(model: Sequential, img_size: tuple, img_path: str) -> None:
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    prediction = model.predict(np.expand_dims(img_array, axis=0))[0][0]
    logging.info(f"Prediction: {'Cat' if prediction < 0.5 else 'Dog'}")

def main(train_from_gui=False):
    global IMG_SIZE
    logging.basicConfig(level=logging.INFO)

    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32
    PATIENCE = 3

    train_datagen, val_datagen, test_datagen = initialize_datagen()

    if train_from_gui:
        model = load_model("trained_model.h5")
    else:
        model = create_model(IMG_SIZE)

    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy', 'Precision', 'Recall']
    )
    
    model.summary()  # Optionally display the model summary

    if train_from_gui:
        train_data = train_datagen.flow_from_directory(
        'cats_dogs_dataset/train', 
        target_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        class_mode='binary'
    )
    val_data = val_datagen.flow_from_directory(
        'cats_dogs_dataset/val', 
        target_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        class_mode='binary'
    )
    test_data = test_datagen.flow_from_directory(
        'cats_dogs_dataset/test', 
        target_size=IMG_SIZE, 
        batch_size=BATCH_SIZE, 
        class_mode='binary'
    )

    callbacks = [
        TensorBoard(log_dir="logs"), 
        EarlyStopping(monitor='val_loss', patience=PATIENCE)
    ]

    if train_from_gui:
        history = model.fit(
            train_data,
            steps_per_epoch=train_data.n // BATCH_SIZE,
            epochs=50,
            validation_data=val_data,
            validation_steps=val_data.n // BATCH_SIZE,
            callbacks=callbacks
    )

    metrics = model.evaluate(test_data, steps=test_data.n // BATCH_SIZE)
    test_loss, test_acc, test_prec, test_recall = metrics
    logging.info(f'Test loss: {test_loss}, Test accuracy: {test_acc}, Test precision: {test_prec}, Test recall: {test_recall}')

    with open('metrics.json', 'w') as f:
        json.dump({
            'loss': test_loss,
            'accuracy': test_acc,
            'precision': test_prec,
            'recall': test_recall
        }, f)

    model.save("trained_model.h5")

    # Initialize Tkinter GUI
    root = Tk()
    my_gui = GUIApp(root, model)
    root.mainloop()

if __name__ == "__main__":
    main()
