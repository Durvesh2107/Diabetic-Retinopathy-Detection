import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os

MODEL_PATH = "model/best_dr_model_compatible.h5"
IMAGE_PATH = "data/train_images/0af296d2f04a.png"
OUTPUT_PATH = "outputs/gradcam_result.png"

class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

os.makedirs("outputs", exist_ok=True)

trained_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    image = cv2.resize(image, (224, 224))
    original_image = image.copy()

    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    return image, original_image

# Rebuild model using logits instead of softmax
base_model = tf.keras.applications.EfficientNetB0(
    weights=None,
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.set_weights(trained_model.layers[0].get_weights())

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)

dense = tf.keras.layers.Dense(128, activation="relu")
x = dense(x)
dense.set_weights(trained_model.layers[2].get_weights())

dropout = tf.keras.layers.Dropout(0.3)
x = dropout(x, training=False)

logits_layer = tf.keras.layers.Dense(5, activation=None)
logits = logits_layer(x)
logits_layer.set_weights(trained_model.layers[4].get_weights())

grad_model = tf.keras.models.Model(
    inputs=base_model.input,
    outputs=[
        base_model.get_layer("top_conv").output,
        logits
    ]
)

processed_image, original_image = preprocess_image(IMAGE_PATH)

with tf.GradientTape() as tape:
    conv_outputs, logits_output = grad_model(processed_image)

    predicted_class = tf.argmax(logits_output[0])
    class_output = logits_output[:, predicted_class]

grads = tape.gradient(class_output, conv_outputs)

print("Grad min:", tf.reduce_min(grads).numpy())
print("Grad max:", tf.reduce_max(grads).numpy())
print("Grad mean:", tf.reduce_mean(grads).numpy())

pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]

heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap).numpy()

heatmap = np.maximum(heatmap, 0)
heatmap = heatmap / (np.max(heatmap) + 1e-8)

heatmap = cv2.resize(
    heatmap,
    (original_image.shape[1], original_image.shape[0])
)

heatmap = np.uint8(255 * heatmap)

heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

overlay = cv2.addWeighted(
    original_image,
    0.6,
    heatmap_color,
    0.4,
    0
)

probs = tf.nn.softmax(logits_output).numpy()
predicted_label = class_names[int(predicted_class)]
confidence = np.max(probs[0]) * 100

print("Predicted Class:", predicted_label)
print("Confidence:", confidence)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(heatmap_color)
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title(f"Overlay: {predicted_label}")
plt.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
plt.show()

print("Grad-CAM saved at:", OUTPUT_PATH)