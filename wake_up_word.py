import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
from tensorflow.python.keras.models import load_model

fs = 44100
seconds = 2
filename = "prediction.wav"
class_names = ["Wake Word NOT Detected", "Wake Word Detected"]

model = load_model("saved_model/WWD.h5")

while True:
    print("Say Now: ")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()
    write(filename, fs, myrecording)

    audio, sample_rate = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_processed = np.mean(mfcc.T, axis=0)

    prediction = model.predict(np.expand_dims(mfcc_processed, axis=0))
    if prediction[:, 1] > 0.99:
        print(f"Wake Word Detected")
        answer = "안녕하세요! 궁금한 것을 물어보세요!"

    else:
        pass