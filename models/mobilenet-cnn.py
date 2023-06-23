import pandas as pd
import numpy as np
import os
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

image_dir = Path('E:/Dataset - dump')
filepaths = list(image_dir.glob(r'**/*.jpg'))
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

images = pd.concat([filepaths, labels], axis=1)
num_class = images.Label.value_counts().count()


train_df, test_df = train_test_split(images, train_size=0.7, shuffle=True, random_state=1)

train_generator = ImageDataGenerator(
    preprocessing_function = tf.keras.applications.mobilenet_v3.preprocess_input,
    validation_split=0.2)

test_generator = ImageDataGenerator(
    preprocessing_function = tf.keras.applications.mobilenet_v3.preprocess_input)


train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)

val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)
pretrained_model = tf.keras.applications.mobilenet.MobileNet(
                    input_shape=(224, 224, 3),
                    include_top=False,
                    weights='imagenet',
                    pooling='avg')
pretrained_model.trainable = False

inputs = pretrained_model.input
x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(50, activation='relu')(x)
outputs = tf.keras.layers.Dense(num_class, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
print(model.summary())



model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)
history = model.fit(train_images,validation_data=val_images,epochs=50,callbacks=[callbacks])



model.save('my_model.h5')
score = model.evaluate(val_images)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


## prediction 

image_path = 'E:/KNN - test/15/msp1.jpg'  # Replace with the actual image path

# Load and preprocess the image
img = image.load_img(image_path, target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = tf.keras.applications.resnet50.preprocess_input(img)

# Make prediction
prediction = model.predict(img)
predicted_label = np.argmax(prediction)
class_mapping = train_images.class_indices  # Get the mapping between class indices and class labels

# Reverse the class mapping
reverse_mapping = {v: k for k, v in class_mapping.items()}

# Get the predicted class label
predicted_class_label = reverse_mapping[predicted_label]

print("Predicted class label:", predicted_class_label)
