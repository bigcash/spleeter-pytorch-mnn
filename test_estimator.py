import torch
import torchaudio
from librosa.core import load
from librosa.output import write_wav
import numpy as np
import time

from spleeter.estimator import Estimator

es = Estimator(2, './pretrained/2stems/model')

time1 = time.time()
for k in range(100):
    # load wav audio
    wav, sr = torchaudio.load_wav('./audio_example.mp3')

    # normalize audio
    wav_torch = wav / (wav.max() + 1e-8)

    wavs = es.separate(wav_torch)
    for i in range(len(wavs)):
        fname = 'out_{}.wav'.format(i)
        print('Writing ', fname)
        write_wav(fname, np.asfortranarray(wavs[i].squeeze().numpy()), sr)
time2 = time.time()
print("cost", time2-time1, (time2-time1)/100)
