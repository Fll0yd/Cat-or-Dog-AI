# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Define directories
train_dir = 'cats_dogs_dataset/train'
val_dir = 'cats_dogs_dataset/val'
test_dir = 'cats_dogs_dataset/test'

# Define data generators with data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define data generators for the validation and test sets
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Define image size and batch size
img_size = (128, 128)
batch_size = 32

# Create data generators for the training, validation, and test sets
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='binary',
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='binary'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='binary'
)

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
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

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Print model summary
model.summary()

#Fit the model on the training data and validate on the validation data
history = model.fit(
train_data,
steps_per_epoch=train_data.n // batch_size,
epochs=50,
validation_data=val_data,
validation_steps=val_data.n // batch_size
)

#Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_data, verbose=0)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

#Load and preprocess a new image for prediction
img_path = 'new_image.jpg'
img = load_img(img_path, target_size=img_size, color_mode='grayscale')
img_array = img_to_array(img) / 255.0
img_array = tf.expand_dims(img_array, axis=0)

#Make a prediction on the new image and print the result
prediction = model.predict(img_array)[0][0]
if prediction < 0.5:
print('Cat')
else:
print('Dog')