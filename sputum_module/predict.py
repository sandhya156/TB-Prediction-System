import sys
import numpy as np
import tensorflow as tf

MODEL_PATH = "sputum_model_finetuned.keras"
model = tf.keras.models.load_model(MODEL_PATH)

def predict_sputum(img_path):
    """Returns TB probability (0 to 1). 1 = TB positive."""
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    arr = tf.keras.utils.img_to_array(img)[None, ...]
    return float(model.predict(arr, verbose=0)[0][0])

if __name__ == "__main__":
    p = predict_sputum(sys.argv[1])
    print(f"TB probability: {p:.3f} -> {'Positive' if p > 0.5 else 'Negative'}")