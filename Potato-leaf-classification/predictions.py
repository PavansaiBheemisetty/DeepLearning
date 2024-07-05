import numpy as np
from PIL import Image
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("potatoes.h5")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def preprocess_image(image_path):
    # Load and preprocess an image
    image = np.array(Image.open(image_path).convert("RGB").resize((256, 256)))
    image = image / 255.0  # Normalize the image to 0-1 range
    return tf.expand_dims(image, 0)  # Expand dimensions to match the input shape

# Load and preprocess the test image
image_path = "download.jpg"
img_array = preprocess_image(image_path)

# Make a prediction
predictions = model.predict(img_array)
predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
confidence = np.max(predictions[0])

print(f"Predicted class: {predicted_class}, Confidence: {confidence:.2f}")

# Evaluate the model on a test set
def evaluate_model(model, test_images, test_labels):
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_classes == test_labels)
    print(f"Accuracy on the test set: {accuracy:.2f}")

# Load your test dataset
# Assuming you have test_images and test_labels as numpy arrays
# test_images = ...
# test_labels = ...
# preprocess the test images as well
# test_images = np.array([preprocess_image(img) for img in test_images])
# test_images = np.vstack(test_images)  # Stack to create the correct shape

# evaluate_model(model, test_images, test_labels)