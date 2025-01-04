import os
import pandas as pd
import numpy as np
import librosa
import glob

def load_and_process_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    return np.log(spectrogram + 1e-9)

def save_spectrograms_to_npy(data_directory, output_csv):
    data = []
    labels = []

    # Process words
    word_files = glob.glob(os.path.join(data_directory, "words", "*.wav"))
    for file in word_files:
        label = os.path.basename(file).split('_')[1]
        spectrogram = load_and_process_audio(file)
        npy_path = f"{file}.npy"
        np.save(npy_path, spectrogram)
        data.append(npy_path)
        labels.append(label)

    # Process sentences
    sentence_files = glob.glob(os.path.join(data_directory, "sentences", "*.wav"))
    for file in sentence_files:
        label = os.path.basename(file).split('_')[1]
        spectrogram = load_and_process_audio(file)
        npy_path = f"{file}.npy"
        np.save(npy_path, spectrogram)
        data.append(npy_path)
        labels.append(label)

    df = pd.DataFrame({'spectrogram_path': data, 'label': labels})
    df.to_csv(output_csv, index=False)
    print(f"Spectrograms saved to {output_csv}.")

# Main Execution
data_directory = "D:\\Uni Work\\Python\\Speech Recognition\\data"
csv_filename = "D:\\Uni Work\\Python\\Speech Recognition\\spectrogram_labels.csv"
save_spectrograms_to_npy(data_directory, csv_filename)
