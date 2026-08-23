import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load the saved model
model = load_model('tb_xray_model.h5')

# Recreate validation data generator (same settings as training)
dataset_path = "dataset/TB_Chest_Radiography_Database"

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

val_gen = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# Get predictions
predictions = model.predict(val_gen)
predicted_classes = (predictions > 0.5).astype(int).flatten()

# True labels
true_classes = val_gen.classes

# Print evaluation report
print("Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=['Normal', 'Tuberculosis']))

print("\nConfusion Matrix:")
print(confusion_matrix(true_classes, predicted_classes))