# Stream audio from microphone
# Classify chunks of audio
###### Importing the libraries #######

import os
import numpy as np
import pyaudio
from scipy.signal import butter, lfilter, hilbert
from pyAudioAnalysis import audioFeatureExtraction
from keras.models import load_model

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

RATE = 44100
CHUNK = int(0.1*RATE)
model = load_model('model3.h5')

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,  
		 frames_per_buffer=CHUNK)

th = 0.5
prob_list = []
count = 0

try:
	while True:
		count += 1
		
		data = np.fromstring(stream.read(CHUNK), dtype=np.int16)
		y = preprocess(data)
		features_list = audioFeatureExtraction.stFeatureExtraction(y, RATE, CHUNK, CHUNK)
		p = model.predict(features_list[0].reshape(1,34), batch_size=None, verbose=0)
		p = p.flatten()
		prob_list.append(p)

		if count%10 == 0:
			prob = np.mean(prob_list)

			if prob >= th:
				print("Ambulance is comming. Set the Green signal quick- {}".format(prob))
			else:
				print("Non Emergency vehicles-{}".format(prob))

			prob_list = [prob]

except KeyboardInterrupt:
	stream.stop_stream()  
	stream.close()  
	p.terminate() 
