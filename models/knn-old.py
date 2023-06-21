import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import glob

# Set the path to your dataset folder
dataset_path = "E:/Dataset - dump/gds"

# Initialize lists to store the image data and corresponding labels
data = []
labels = []

# Iterate over subfolders in the dataset path
for folder_path in glob.glob(os.path.join(dataset_path, "*")):
    if os.path.isdir(folder_path):
        label = os.path.basename(folder_path)

        # Iterate over images in the subfolder
        for image_path in glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png")):
            # Read the image using OpenCV
            image = cv2.imread(image_path)
            if image is not None:
                # Convert the image to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Thresholding / Binarization
                ret, imgf = cv2.threshold(gray_image, 110, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                # Resize the image to a consistent size using OpenCV
                resized_image = cv2.resize(imgf, (32, 32))
                # Convert the resized image to a numpy array
                image_array = np.array(resized_image).flatten()
                # Append the image array and label to the data and labels lists
                data.append(image_array)
                labels.append(label)

# Convert the data and labels lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Read and preprocess your own image
image_path = "E:/KNN - test/1.jpg"  # Replace with the path to your image
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, imgf = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

resized_image = cv2.resize(imgf, (32, 32))
image_array = np.array(resized_image).flatten()

# Make predictions on your image
prediction = knn.predict([image_array])

# Print the predicted label
print("Predicted label:", prediction)
