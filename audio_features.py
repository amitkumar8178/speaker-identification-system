# audio_features.py

import numpy as np
import librosa

SAMPLE_RATE = 16000
DURATION = 3
N_MFCC = 40
N_MELS = 128
N_FFT = 512
HOP_LENGTH = 256

def load_and_preprocess_audio(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    audio, _ = librosa.effects.trim(audio)
    if len(audio) > SAMPLE_RATE * DURATION:
        audio = audio[:SAMPLE_RATE * DURATION]
    else:
        audio = np.pad(audio, (0, SAMPLE_RATE * DURATION - len(audio)))
    return audio

def extract_features(audio):
    mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    features = np.concatenate([mfccs, delta_mfccs, delta2_mfccs, mel_spectrogram])

    
    target_length = SAMPLE_RATE * DURATION // HOP_LENGTH + 1
    if features.shape[1] < target_length:
        features = np.pad(features, ((0, 0), (0, target_length - features.shape[1])))
    else:
        features = features[:, :target_length]

    return features


def prepare_data(file_path):
    X = []

    
    audio = load_and_preprocess_audio(file_path)
    features = extract_features(audio)
    X.append(features)

    X_test = np.array(X)


    return X_test

