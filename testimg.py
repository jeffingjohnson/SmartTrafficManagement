import urllib.request
import numpy as np
while True:
    with urllib.request.urlopen('http://192.168.137.211') as f:
        audio=f.read().split()
        print(audio)
        #audio=audio.split()
        #print(audio)
        data = np.array(audio, dtype=np.int16)
        print(data)
       
        print(len(data))
