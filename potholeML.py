import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the input shape for the images
input_shape = (256, 256, 3)

# Define the CNN model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create an image data generator for training and validation
train_datagen = ImageDataGenerator(rescale=1.0/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
valid_datagen = ImageDataGenerator(rescale=1.0/255)

# Load the training and validation datasets from directories
train_dir = "C:/Users/Admin/Downloads/data/data/training"  # Update with the actual path
valid_dir = "C:/Users/Admin/Downloads\data/data/validation"  # Update with the actual path
train_dataset = train_datagen.flow_from_directory(train_dir, target_size=input_shape[:2], batch_size=32, class_mode='binary')
valid_dataset = valid_datagen.flow_from_directory(valid_dir, target_size=input_shape[:2], batch_size=32, class_mode='binary')

# Train the model
history = model.fit(train_dataset, validation_data=valid_dataset, epochs=10)

# Save the trained model
model.save('pothole_detector.h5')
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
loaded_model = load_model('pothole_detector.h5')

# Preprocess the new image
#image_path = '"C:/Users/Admin/Downloads/data/data/non-potholes/test/28.jpg"'
image_path = 'C:/Users/Admin/Downloads/data/data/non-potholes/test/28.jpg'

new_image = load_img(image_path, target_size=(256, 256))
new_image = img_to_array(new_image)
new_image = np.expand_dims(new_image, axis=0)
new_image /= 255.0

# Make a prediction
prediction = loaded_model.predict(new_image)

# Interpret the prediction
if prediction >= 0.5:
    print("Pothole")
else:
    print("Not a pothole.") 