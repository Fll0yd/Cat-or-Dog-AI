import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

train_dir = 'cats_dogs_dataset/train'
val_dir = 'cats_dogs_dataset/val'
test_dir = 'cats_dogs_dataset/test'

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

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

img_size = (128, 128)
batch_size = 32

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='binary'
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

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n//batch_size,
    epochs=50,
    validation_data=val_generator,
    validation_steps=val_generator.n//batch_size
)

test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.n//batch_size)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

img_path = 'new_image.jpg'
img = image.load_img(img_path, target_size=(128, 128), grayscale=True)

img_array = image.img_to_array(img)
img_array /= 255.

prediction = model.predict(np.expand_dims(img_array, axis=0))[0][0]
if prediction < 0.5:
    print('Cat')
else:
    print('Dog')

test_data = test_generator[0][0]
test_labels = test_generator[0][1]
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc)
