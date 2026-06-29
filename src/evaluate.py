import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model("model/best_model.keras")
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Loss:", test_loss)
print("Accuracy:", test_accuracy)

# You can load saved test data here if needed

print("Evaluation Complete")