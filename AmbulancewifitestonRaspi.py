# Stream audio from microphone
# Classify chunks of audio
###### Importing the libraries #######

import os
import numpy as np
import pyaudio
from scipy.signal import butter, lfilter, hilbert,resample
from pyAudioAnalysis import audioFeatureExtraction
from keras.models import load_model
import urllib.request
import RPi.GPIO as gpio

gpio.setmode(gpio.BCM)
gpio.setup(18, gpio.OUT)
###### Data Pre-processing functions ###

def butter_bandpass_filter(data, lowcut=500, highcut=1500, fs=44100, order=5):
		nyq = 0.5 * fs
		low = lowcut / nyq
		high = highcut / nyq

		b, a = butter(order, [low, high], btype='band')
		y = lfilter(b, a, data)
		return y

def preprocess(y):
		y_filt = butter_bandpass_filter(y)
		analytic_signal = hilbert(y_filt)
		amplitude_envelope = np.abs(analytic_signal)
		return amplitude_envelope

#######################################

RATE = 5000
#CHUNK = int(0.1*RATE)
CHUNK=4410
model = load_model('Trained_data.h5')
gpio.setmode(gpio.BCM)
gpio.setup(18, gpio.OUT)

#p = pyaudio.PyAudio()
#stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,  input_device_index=0, frames_per_buffer=CHUNK)

th = 0.3
prob_list = []
count = 0
var=0
try:
    while True:
            count += 1
            with urllib.request.urlopen('http://192.168.137.204') as f:
                audio=f.read().split()
                data = np.array(audio, dtype=np.int16)
            number_of_samples = round(len(data) * float(44100) / RATE)
            data = resample(data, number_of_samples)
            y = preprocess(data)
            features_list = audioFeatureExtraction.stFeatureExtraction(y, 44100, number_of_samples, number_of_samples)
            #features_list = audioFeatureExtraction.stFeatureExtraction(y, RATE, CHUNK, CHUNK)
            p = model.predict(features_list[0].reshape(1,34), batch_size=None, verbose=0)
            p = p.flatten()
            prob_list.append(p)

            if count%1 == 0:
                    prob = np.mean(prob_list)

                    if prob >= th:
                            print("Ambulance is coming. Set the Green signal quick- {}".format(prob))
                            #print("Ambulance is coming. Set the Green signal quick")
                            var += 1
                        
                    else:
                            print("Non Emergency vehicles-{}".format(prob))
                            #print("Non Emergency vehicles
                            var = 0

                    prob_list = [prob]
                    if var > 2:
                        gpio.output(18, gpio.HIGH)
                    else:
                        gpio.output(18, gpio.LOW)

except KeyboardInterrupt:
	#stream.stop_stream()  
	#stream.close()  
	p.terminate()
