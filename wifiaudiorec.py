import pyaudio
import wave
import urllib.request
import numpy as np
chunk = 8820  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 5749  # Record at 44100 samples per second
seconds = 15
filename = "output.wav"
p = pyaudio.PyAudio()
frames = []  # Initialize array to store frames

# Store data in chunks for 3 seconds
for i in range(0, int(fs / chunk * seconds)):
    with urllib.request.urlopen('http://192.168.137.47') as f:
        audio=f.read().split()
        d = np.array(audio, dtype=np.int16)
        data=d.tobytes()
    frames.append(data)

# Stop and close the stream

# Terminate the PortAudio interface

print('Finished recording')

# Save the recorded data as a WAV file
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()

