from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 
import os
import cv2
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint,
    ReduceLROnPlateau
)

os.makedirs("model", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

df=pd.read_csv("data/train.csv")
print(df["diagnosis"].value_counts())
df["id_code"] = df["id_code"].astype(str) + ".png"
df["diagnosis"] = df["diagnosis"].astype(str)


#Train Test Split
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["diagnosis"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["diagnosis"]
)

print("Training Images:", len(train_df))
print("Testing Images:", len(test_df))

#Creating processing function
def apply_clahe(image):
    image = image.astype(np.uint8)
    lab=cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l,a,b=cv2.split(lab)

    clahe=cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )
    l=clahe.apply(l)
    lab=cv2.merge((l,a,b))
    image=cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
    return image

def preprocess_image(image):
    #CLAHE
    image = image.astype(np.uint8)
    image=apply_clahe(image)

    #Noise reduction
    image=cv2.GaussianBlur(
        image,
        (3,3),
        0
    )

    #Sharpening 
    kernel=np.array([
        [-1,-1,-1],
        [-1,9,-1],
        [-1,-1,-1]
    ])
    image = cv2.filter2D(
        image, 
        -1, 
        kernel
        )
    image = image.astype("float32") / 255.0
    return image


#Image Generator
# Training Generator
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3]
)

# Validation and Test Generator
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image,
)

#Train Generator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory="data/train_images",
    x_col="id_code",
    y_col="diagnosis",
    target_size=(224, 224),
    batch_size=4,
    class_mode="categorical",
    shuffle=True
)

#Test Generator
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory="data/train_images",
    x_col="id_code",
    y_col="diagnosis",
    target_size=(224, 224),
    batch_size=4,
    class_mode="categorical",
    shuffle=False
)
validation_generator = test_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory="data/train_images",
    x_col="id_code",
    y_col="diagnosis",
    target_size=(224, 224),
    batch_size=4,
    class_mode="categorical",
    shuffle=False
)

#Load EfficientNet
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)


base_model.trainable = True
for layer in base_model.layers[:30]:
    layer.trainable=False

model = tf.keras.models.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(
        128,
        activation='relu'
    ),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(
        5,
        activation='softmax'
    )
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


#Model Summary
model.summary()

early_stop=EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

checkpoint=ModelCheckpoint(
    "model/best_model.keras",
    save_best_only=True
)

reduce_lr=ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2, 
    verbose=1
)
#Class Weights
class_weights= compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_df["diagnosis"]),
    y=train_df["diagnosis"]
)
class_weights=dict(enumerate(class_weights))
print("Class Weights", class_weights)


history=model.fit(
    train_generator,
    validation_data=(validation_generator),
    epochs=20,
    class_weight=class_weights,
    callbacks=[early_stop,checkpoint, reduce_lr]
)

#Evaluate Model
test_loss,test_accuracy=model.evaluate(test_generator)
print("\nTest Accuracy:", test_accuracy)

from sklearn.metrics import classification_report
import numpy as np

# Predictions
y_pred = model.predict(test_generator)

# Convert probabilities to class labels
y_pred_classes = np.argmax(y_pred, axis=1)

# True labels
y_true = test_generator.classes

class_names=[
    "No DR", 
    "Mild", 
    "Moderate", 
    "Severe", 
    "Proliferative DR"
]

# Report
print(classification_report(
    y_true,
    y_pred_classes,
    labels=[0, 1, 2, 3, 4],
    target_names=class_names,
    zero_division=0
))
# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes, labels=[0, 1, 2, 3, 4])

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.show()

model.save("model/best_model.keras")
print("Model saved Successfully!")
