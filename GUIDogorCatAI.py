import os
import numpy as np
import tensorflow as tf
import logging
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image as pil_image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tkinter import Tk, Button, filedialog

import argparse

class ImageClassifier:
    def __init__(self, img_size=(128, 128), batch_size=32, patience=3):
        self.IMG_SIZE = img_size
        self.BATCH_SIZE = batch_size
        self.PATIENCE = patience
        self.EPOCHS = 10  # Replace 10 with the desired number of epochs
        self.script_directory = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = os.path.join(self.script_directory, 'f:/Code/Python/CatorDog/cats_dogs_dataset')
        self.TRAIN_DIR = os.path.join(self.dataset_path, 'train')
        self.VAL_DIR = os.path.join(self.dataset_path, 'val')
        self.TEST_DIR = os.path.join(self.dataset_path, 'test')
        self.log_dir = os.path.join(self.script_directory, 'logs')
        self.model_path = os.path.join(self.script_directory, 'Python', 'CatorDog', 'trained_model.h5')
        self.TRAIN_SIZE = 25000  # Replace this with the actual number of training samples
        self.VALIDATION_SIZE = 50222  # Replace this with the actual number of validation samples

    def initialize_datagen(self, with_augmentation=True):
        if with_augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1. / 255)

        val_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        return train_datagen, val_datagen, test_datagen

    def create_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.IMG_SIZE, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        return model

    def on_train_button_click(self):
        logging.info("Training initiated from GUI.")

        # Specify the correct log directory path
        log_dir = self.log_dir

        train_data, val_data, test_data = self.initialize_data_generators()

        model = self.create_model()
        callbacks = [TensorBoard(log_dir=log_dir),
                    EarlyStopping(monitor='val_loss', patience=self.PATIENCE)]

        model.compile(optimizer='adam', loss='binary_crossentropy',
                    metrics=['accuracy', tf.keras.metrics.Precision(),
                            tf.keras.metrics.Recall()])

        model.summary()

        steps_per_epoch = self.TRAIN_SIZE // self.BATCH_SIZE  # Fix here
        validation_steps = self.VALIDATION_SIZE // self.BATCH_SIZE  # Fix here

        history = model.fit(
            train_data,
            steps_per_epoch=steps_per_epoch,
            epochs=self.EPOCHS,
            validation_data=val_data,
            validation_steps=validation_steps,
            callbacks=callbacks
        )

        self.plot_metrics({
            'accuracy': history.history['accuracy'][-1],
            'precision': history.history['precision'][-1],
            'recall': history.history['recall'][-1],
            'val_accuracy': history.history['val_accuracy'][-1],
            'val_precision': history.history['val_precision'][-1],
            'val_recall': history.history['val_recall'][-1],
            'loss': history.history['loss'][-1],
            'val_loss': history.history['val_loss'][-1]
        })

        if test_data.samples > 0:
            metrics = model.evaluate(test_data, steps=len(test_data))
            test_loss, test_acc, test_precision, test_recall = metrics
            logging.info(f'Test loss: {test_loss}, Test accuracy: {test_acc}, '
                        f'Test precision: {test_precision}, Test recall: {test_recall}')
            with open('metrics.json', 'w') as f:
                json.dump({
                    'loss': test_loss,
                    'accuracy': test_acc,
                    'precision': test_precision,
                    'recall': test_recall
                }, f)
        else:
            logging.warning("No test samples found.")

        model.save(os.path.join(self.script_directory, 'Python', 'CatorDog', 'trained_model.h5'))

    def setup_gui(self):
        root = Tk()
        root.title("Image Classifier GUI")

        train_button = Button(root, text="Train Model", command=self.on_train_button_click)
        train_button.pack()

        predict_button = Button(root, text="Predict Image", command=self.predict_image)
        predict_button.pack()

        plot_metrics_button = Button(root, text="Plot Metrics", command=self.plot_saved_metrics)
        plot_metrics_button.pack()

        root.mainloop()

    def main(self):
        self.setup_gui()

    def initialize_data_generators(self, with_augmentation=True):
        train_datagen, val_datagen, test_datagen = self.initialize_datagen(with_augmentation)
        train_data = train_datagen.flow_from_directory(
            self.TRAIN_DIR, target_size=self.IMG_SIZE, batch_size=self.BATCH_SIZE, class_mode='binary'
        )
        val_data = val_datagen.flow_from_directory(
            self.VAL_DIR, target_size=self.IMG_SIZE, batch_size=self.BATCH_SIZE, class_mode='binary'
        )
        test_data = test_datagen.flow_from_directory(
            self.TEST_DIR, target_size=self.IMG_SIZE, batch_size=self.BATCH_SIZE, class_mode='binary', shuffle=False
        )

        # Add try-except block to handle UnidentifiedImageError
        def safe_flow_from_directory(generator):
            while True:
                try:
                    yield next(generator)
                except pil_image.UnidentifiedImageError:
                    continue

        train_data = safe_flow_from_directory(train_data)
        val_data = safe_flow_from_directory(val_data)
        test_data = safe_flow_from_directory(test_data)

        return train_data, val_data, test_data

    def plot_metrics(self, metrics):
        metrics_names = ['accuracy', 'precision', 'recall', 'val_accuracy', 'val_precision', 'val_recall', 'loss', 'val_loss']
        plt.figure(figsize=(10, 6))
        plt.bar(metrics_names, [metrics[name] for name in metrics_names], color=['blue', 'green', 'orange', 'red'])
        plt.ylabel('Metrics Value')
        plt.title('Test Metrics')
        plt.show()

    def predict_image(self):
        self.model_path = os.path.join(self.script_directory, 'Python', 'CatorDog', 'trained_model.h5')
        model = load_model(self.model_path)
        file_path = filedialog.askopenfilename(title="Select an image for prediction", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.predict_single_image(model, file_path)

    def predict_single_image(self, model, img_path):
        img = image.load_img(img_path, target_size=self.IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        result = "Cat" if prediction[0][0] < 0.5 else "Dog"

        # Display the image
        img.show()

        # Display the prediction result
        messagebox.showinfo("Prediction Result", f"The model predicts that the image contains a {result}.")

    def plot_saved_metrics(self):
        try:
            with open('metrics.json', 'r') as f:
                metrics = json.load(f)
                self.plot_metrics(metrics)
        except FileNotFoundError:
            logging.warning("No saved metrics file found.")

# Add if __name__ == "__main__": block to allow running from both GUI and command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an image classifier model.')
    parser.add_argument('--train', action='store_true', help='Train the model')
    args = parser.parse_args()

    classifier = ImageClassifier()

    if args.train:
        classifier.on_train_button_click()
    else:
        classifier.main()
