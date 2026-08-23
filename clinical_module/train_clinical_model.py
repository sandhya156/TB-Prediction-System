import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
import joblib

# Load dataset (train.csv has the target column, so we split it ourselves)
df = pd.read_csv('dataset2/train.csv')

# Features (excluding xray_result to avoid data leakage, and tb_diagnosis which is target)
features = ['patient_age', 'gender', 'cough_duration', 'fever', 'weight_loss',
            'night_sweats', 'chest_pain', 'shortness_breath', 'hemoptysis',
            'smoking_status', 'bmi', 'contact_tb']

target = 'tb_diagnosis'

X = df[features]
y = df[target]

# Split into train/test ourselves
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'clinical_scaler.pkl')

# Save feature list for later use in dashboard
joblib.dump(features, 'clinical_features.pkl')

# Handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# Build ANN model
model = models.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=16,
    class_weight=class_weights
)

# Evaluate
predictions = (model.predict(X_test_scaled) > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=['Normal', 'Tuberculosis']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save model
model.save('tb_clinical_model.h5')
print("\nClinical model saved successfully!")
