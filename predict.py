import torch
import numpy as np
import joblib
import os
from audio_features import extract_features, load_and_preprocess_audio, prepare_data

MODEL_PATH = r"C:\Users\91892\OneDrive\Documents\speaker identification system\student_model_distilled.pth"
ENCODER_PATH = r"C:\Users\91892\OneDrive\Documents\speaker identification system\label_encoder.pkl"

label_encoder = joblib.load(ENCODER_PATH)

from model_architecture import StudentCNN

NUM_SPEAKERS = len(label_encoder.classes_) 
INPUT_SHAPE =(248, 188)
model = StudentCNN(num_speakers=NUM_SPEAKERS, input_shape=INPUT_SHAPE)
state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.eval()

def predict_speaker(audio_path):
    audio_name = os.path.basename(audio_path)

    print(f"\n🔍 Audio Processing Pipeline for 🎧 '{audio_name}':")
    print(f"{audio_name}")
    print("    ↓")
    print(f"librosa.load('{audio_name}') → waveform (y)")
    print("    ↓")
    print("Feature Extraction (MFCC, delta, etc.)")
    print("    ↓")
    print(f"Feature Tensor → SpeakerCNN Model")
    print("    ↓")
    print("Conv Layers → FC Layers → Logits")
    print("    ↓")
    print("Softmax → Class Probabilities")
    print("    ↓")
    print("Argmax → Predicted Speaker")
    
    xtest = prepare_data(audio_path)
    xtest = np.array(xtest)
    xtest_tensor = torch.tensor(xtest, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    if xtest_tensor.ndim == 5:
        xtest_tensor = xtest_tensor.squeeze(0)


    with torch.no_grad():
        outputs = model(xtest_tensor)
        predicted_label = torch.argmax(outputs, dim=1).item()

    speaker_name = label_encoder.inverse_transform([predicted_label])[0]
    print(speaker_name)
    return speaker_name




