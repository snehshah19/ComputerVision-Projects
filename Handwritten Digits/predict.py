import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import cv2

# Create a blank white image (28x28)
image = np.ones((28, 28), dtype=np.uint8) * 255

# Draw a handwritten digit "2" on the image
image = cv2.putText(image, "2", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

# Invert the colors (MNIST format: black background, white digit)
image = cv2.bitwise_not(image)

# Save the image
cv2.imwrite("sample_digit_2.png", image)

# Display the image
cv2.imshow("Sample Digit 2",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Load the sample image
sample_image_path = "sample_digit_2.png"  # Path to the sample image
img = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)

# Display the sample image
plt.imshow(img, cmap='gray')
plt.title("Sample Input Image")
plt.show()


# load the model
model = keras.models.load_model("model.h5")

# Preprocess the sample image
img = cv2.resize(img, (28, 28))  # Resize to 28x28
img = img / 255.0  # Normalize
img = img.reshape(1, 28, 28, 1)  # Reshape for the model

# Make prediction
prediction = model.predict(img)
predicted_digit = np.argmax(prediction)  # Get the class with the highest probability

print(f"Predicted Digit: {predicted_digit}")
print(f"Prediction Probabilities: {prediction}")