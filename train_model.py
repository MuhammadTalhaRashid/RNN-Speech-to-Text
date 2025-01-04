import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import regularizers  # Import regularizers module

def get_max_timesteps(csv_filename):
    df = pd.read_csv(csv_filename)
    max_timesteps = 0
    
    # Loop through all the spectrograms in the CSV file
    for path in df['spectrogram_path']:
        spectrogram = np.load(path)  # Load .npy file
        # Check the second dimension (time steps) of the spectrogram
        max_timesteps = max(max_timesteps, spectrogram.shape[1])
    
    return max_timesteps

def load_data(csv_filename, max_timesteps):
    df = pd.read_csv(csv_filename)
    X = []
    
    for path in df['spectrogram_path']:
        spectrogram = np.load(path)  # Load .npy file
        
        # Truncate or pad the spectrogram to the max_timesteps
        if spectrogram.shape[1] > max_timesteps:
            spectrogram = spectrogram[:, :max_timesteps]  # Truncate time steps
        elif spectrogram.shape[1] < max_timesteps:
            # Pad with zeros if the spectrogram has fewer time steps
            padding = max_timesteps - spectrogram.shape[1]
            spectrogram = np.pad(spectrogram, ((0, 0), (0, padding)), mode='constant', constant_values=0)
        
        X.append(spectrogram)
    
    y = df['label'].values
    return np.array(X), y

def build_rnn_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        
        # Adding L2 regularization to the LSTM layers
        tf.keras.layers.LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001)),  # L2 regularization added here
        tf.keras.layers.LSTM(64, kernel_regularizer=regularizers.l2(0.001)),  # L2 regularization added here
        
        # Adding L2 regularization to Dense layers
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  # L2 regularization added here
        
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load data
csv_filename = "D:\\Uni Work\\Python\\Speech Recognition\\spectrogram_labels.csv"

# Step 1: Calculate max time steps across all spectrograms
max_timesteps = get_max_timesteps(csv_filename)
print(f"Max time steps: {max_timesteps}")
print(' ')

# Step 2: Load data and truncate/pad to the max time steps
X, y = load_data(csv_filename, max_timesteps)

# Reshape data for compatibility with RNNs (ensure the correct 3D shape for LSTM)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])  # Shape (samples, timesteps, features)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build and train model
input_shape = X_train.shape[1:]  # Automatically adapt to spectrogram shape
num_classes = len(label_encoder.classes_)
model = build_rnn_model(input_shape, num_classes)

model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

# Save model and label encoder
model.save('D:\\Uni Work\\Python\\Speech Recognition\\my_model.keras')
np.save('D:\\Uni Work\\Python\\Speech Recognition\\label_encoder_classes.npy', label_encoder.classes_)
print("Training complete and model saved.")

# After training, calculate training and validation errors
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

# Display training and cross-validation errors
print(f"Training Error (J_train): {train_loss}")
print(f"Cross-Validation Error (J_val): {val_loss}")
