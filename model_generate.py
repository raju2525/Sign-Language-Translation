import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# Define directory containing sign folders
data_dir = "Data"

# Define target image size
target_image_size = (500,500)

# Preprocess and load images
batch_size = 100

def preprocess_image(img):
    img_array = keras.preprocessing.image.img_to_array(img)
    resized_img = keras.preprocessing.image.smart_resize(img_array, target_image_size)
    return resized_img

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False,
    validation_split=0.2,
    preprocessing_function=preprocess_image  # Using the custom preprocessing function
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=target_image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=target_image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)
print("Number of training samples:", train_generator.samples)
print("Number of validation samples:", validation_generator.samples)

# Define model architecture
model = Sequential([
    Conv2D(75, (3, 3), strides=1, padding='same', activation='relu', input_shape=(target_image_size[0], target_image_size[1], 3)),
    BatchNormalization(),
    MaxPool2D((2, 2), strides=2, padding='same'),
    Conv2D(50, (3, 3), strides=1, padding='same', activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    MaxPool2D((2, 2), strides=2, padding='same'),
    Conv2D(25, (3, 3), strides=1, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPool2D((2, 2), strides=2, padding='same'),
    Flatten(),
    Dense(units=512, activation='relu'),
    Dropout(0.3),
    Dense(units=2, activation='softmax')  # Adjust the number of classes to match your dataset
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)]
)

# Save model
model.save('smnist_with_images_resized.h5')
