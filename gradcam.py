import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

MODEL_PATH = "deepshield_model.keras"
IMAGE_PATH = "test_image.jpg"

IMG_SIZE = 128
LAST_CONV_LAYER = "out_relu"


# Load model
model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# Load image
image = Image.open(IMAGE_PATH).convert("RGB")
original = np.array(image)

img = image.resize((IMG_SIZE, IMG_SIZE))
img_array = np.array(img).astype("float32") / 255.0
img_array = np.expand_dims(img_array, axis=0)


# Find convolution layer
last_conv_layer = model.get_layer(LAST_CONV_LAYER)

print("Grad-CAM layer:", last_conv_layer.name)


# Create Grad-CAM model
grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[
        last_conv_layer.output,
        model.output
    ]
)


# Prediction + gradients
with tf.GradientTape() as tape:

    conv_outputs, prediction = grad_model(img_array)

    real_probability = prediction[:, 0]

    if float(real_probability[0]) >= 0.5:
        class_name = "REAL"
        class_score = real_probability
    else:
        class_name = "AI"
        class_score = 1.0 - real_probability

    print("Prediction:", class_name)
    print("Real probability:", float(real_probability[0]))
    print("AI probability:", float(1.0 - real_probability[0]))


# Gradients
grads = tape.gradient(class_score, conv_outputs)

# Average gradients across spatial dimensions
pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

conv_outputs = conv_outputs[0]
pooled_grads = pooled_grads[0]

# Weight feature maps
heatmap = tf.reduce_sum(
    conv_outputs * pooled_grads,
    axis=-1
)

# ReLU
heatmap = tf.maximum(heatmap, 0)

# Normalize
max_value = tf.reduce_max(heatmap)

if float(max_value) != 0:
    heatmap /= max_value

heatmap = heatmap.numpy()


# Resize heatmap to original image
heatmap = cv2.resize(
    heatmap,
    (original.shape[1], original.shape[0])
)

# Convert to 0-255
heatmap_uint8 = np.uint8(255 * heatmap)

# Apply color map
heatmap_color = cv2.applyColorMap(
    heatmap_uint8,
    cv2.COLORMAP_JET
)

# Convert original RGB -> BGR
original_bgr = cv2.cvtColor(
    original,
    cv2.COLOR_RGB2BGR
)

# Overlay
overlay = cv2.addWeighted(
    original_bgr,
    0.6,
    heatmap_color,
    0.4,
    0
)


# Save results
cv2.imwrite(
    "gradcam_heatmap.jpg",
    heatmap_color
)

cv2.imwrite(
    "gradcam_result.jpg",
    overlay
)

print()
print("================================")
print("GRAD-CAM COMPLETE")
print("================================")
print("Prediction :", class_name)
print("Real       :", round(float(real_probability[0]) * 100, 2), "%")
print("AI         :", round(float(1.0 - real_probability[0]) * 100, 2), "%")
print()
print("Saved:")
print("gradcam_heatmap.jpg")
print("gradcam_result.jpg")