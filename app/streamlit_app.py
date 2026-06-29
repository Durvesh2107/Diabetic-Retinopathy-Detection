import streamlit as st
import tensorflow as tf
from PIL import Image 
import numpy as np 
import cv2 

st.title("Diabetic Retinopathy Detection")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/best_dr_model_compatible.h5",
        compile=False
    )
model=load_model()

class_names=[
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

def apply_clahe(image):
    image = image.astype(np.uint8)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l,a,b))

    image = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )
    return image

def preprocess_image(image):
    image = image.convert("RGB")
    image = np.array(image)

    image = cv2.resize(image, (224, 224))

    image = image.astype("float32") / 255.0

    image = np.expand_dims(image, axis=0)

    return image

# ==========================
# FILE UPLOAD STARTS HERE
# ==========================

uploaded_file = st.file_uploader(
    "Upload Retinal Image",
    type=["jpg","jpeg","png"]
)


if uploaded_file is not None:
    image=Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    processed_image = preprocess_image(image)

    if st.button("Predict"):
        prediction = model.predict(processed_image)
        # Smooth probabilities for display
        epsilon = 0.01

        display_prediction = prediction + epsilon

        display_prediction = (
        display_prediction /
        np.sum(display_prediction)
        )
        st.write(display_prediction)

        predicted_class = np.argmax(prediction)

        confidence = np.max(display_prediction) * 100

        st.success(
            f"Prediction: {class_names[predicted_class]}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )
        st.subheader("Class Probabilities")

        for i, prob in enumerate(display_prediction[0]):

            st.write(
                f"{class_names[i]}: {prob*100:.2f}%"
            )

            st.progress(float(prob))

