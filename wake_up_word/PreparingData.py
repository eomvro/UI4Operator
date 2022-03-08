import sounddevice as sd
from scipy.io.wavfile import write

def record_audio_and_save(save_path, n_times=150):
    input("to start audio_data recording press enter: ")
    for i in range(n_times):
        fs = 44100
        seconds = 2
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()
        write(save_path + str(i) + '.wav', fs, myrecording)
        input(f'press to record next or two stop press ctrl + C ({i+1}/{n_times})')

def record_background_save(save_path, n_times=150):
    input("to start your background_sound sounds press enter: ")
    for i in range(n_times):
        fs = 44100
        seconds = 2
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()
        write(save_path + str(i) + '.wav', fs, myrecording)
        print(f'Currently on {i}/{n_times}')

##record name
#print('recording the wake word \n')
#record_audio_and_save("audio_data/", n_times=150)

##record background
print('recording the background sound \n')
record_audio_and_save("background_sound/", n_times=150)