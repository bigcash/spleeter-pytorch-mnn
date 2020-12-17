# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.09.09
""" python demo usage about MNN API """
from __future__ import print_function
import numpy as np
import MNN
import cv2
import torch
import math
import time
import torchaudio
from torchaudio.functional import istft
import torch.nn.functional as FUNC
from librosa.output import write_wav

F = 1024
T = 512
win_length = 4096
hop_length = 1024
win = torch.hann_window(win_length)


def pad_and_partition(tensor, T):
    """
    pads zero and partition tensor into segments of length T

    Args:
        tensor(Tensor): BxCxFxL

    Returns:
        tensor of size (B*[L/T] x C x F x T)
    """
    old_size = tensor.size(3)
    new_size = math.ceil(old_size/T) * T
    tensor = FUNC.pad(tensor, [0, new_size - old_size])
    [b, c, t, f] = tensor.shape
    split = new_size // T
    return torch.cat(torch.split(tensor, T, dim=3), dim=0)



def compute_stft(wav):
    """
    Computes stft feature from wav

    Args:
        wav (Tensor): B x L
    """
    stft = torch.stft(
        wav, win_length, hop_length=hop_length, window=win)

    # only keep freqs smaller than self.F
    stft = stft[:, :F, :, :]
    real = stft[:, :, :, 0]
    im = stft[:, :, :, 1]
    mag = torch.sqrt(real ** 2 + im ** 2)
    return stft, mag


def inverse_stft(stft):
    """Inverses stft to wave form"""

    pad = win_length // 2 + 1 - stft.size(1)
    stft = FUNC.pad(stft, (0, 0, 0, 0, 0, pad))
    wav = istft(stft, win_length, hop_length=hop_length,
                window=win)
    return wav.detach()

def inference():
    time_m = 0
    time_f = 0
    time_s = time.time()
    """ inference mobilenet_v1 using a specific picture """
    interpreter = MNN.Interpreter("vocals.mnn")
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)

    interpreter2 = MNN.Interpreter("accompaniment.mnn")
    session2 = interpreter2.createSession()
    input_tensor2 = interpreter2.getSessionInput(session2)
    time_m += time.time() - time_s

    time_s = time.time()
    # load wav audio
    wav, sr = torchaudio.load_wav('./audio_example.mp3')
    # normalize audio
    wav_torch = wav / (wav.max() + 1e-8)
    # stft - 2 X F x L x 2
    # stft_mag - 2 X F x L
    stft, stft_mag = compute_stft(wav_torch)
    L = stft.size(2)
    # 1 x 2 x F x T
    stft_mag = stft_mag.unsqueeze(-1).permute([3, 0, 1, 2])
    stft_mag = pad_and_partition(stft_mag, T)  # B x 2 x F x T
    stft_mag = stft_mag.transpose(2, 3)  # B x 2 x T x F
    print(type(stft_mag))
    stft_mag = stft_mag.numpy()
    time_f += time.time() - time_s

    time_s = time.time()
    tmp_input = MNN.Tensor((1, 2, 512, 1024), MNN.Halide_Type_Float, stft_mag, MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)
    output_tensor = interpreter.getSessionOutput(session)
    #constuct a tmp tensor and copy/convert in case output_tensor is nc4hw4
    tmp_output = MNN.Tensor((1, 2, 512, 1024), MNN.Halide_Type_Float, np.ones([1, 2, 512, 1024]).astype(np.float32), MNN.Tensor_DimensionType_Caffe)
    output_tensor.copyToHostTensor(tmp_output)

    input_tensor2.copyFrom(tmp_input)
    interpreter2.runSession(session2)
    output_tensor2 = interpreter2.getSessionOutput(session2)
    # constuct a tmp tensor and copy/convert in case output_tensor is nc4hw4
    tmp_output2 = MNN.Tensor((1, 2, 512, 1024), MNN.Halide_Type_Float, np.ones([1, 2, 512, 1024]).astype(np.float32),
                            MNN.Tensor_DimensionType_Caffe)
    output_tensor2.copyToHostTensor(tmp_output2)

    masks = [torch.from_numpy(tmp_output.getData()), torch.from_numpy(tmp_output2.getData())]
    time_m += time.time() - time_s

    time_s = time.time()
    # compute denominator
    mask_sum = sum([m ** 2 for m in masks])
    mask_sum += 1e-10

    wavs = []
    for mask in masks:
        mask = (mask ** 2 + 1e-10 / 2) / (mask_sum)
        mask = mask.transpose(2, 3)  # B x 2 X F x T

        mask = torch.cat(torch.split(mask, 1, dim=0), dim=3)
        # mask = np.concatenate(np.split(mask, 1, axis=0), axis=3)

        mask = mask.squeeze(0)[:, :, :L].unsqueeze(-1)  # 2 x F x L x 1
        stft_masked = stft * mask
        wav_masked = inverse_stft(stft_masked)

        wavs.append(wav_masked)
    time_f += time.time() - time_s

    write_wav('out_0.wav', np.asfortranarray(wavs[0].squeeze().numpy()), sr)
    write_wav('out_1.wav', np.asfortranarray(wavs[1].squeeze().numpy()), sr)
    return time_m, time_f

if __name__ == "__main__":
    time1 = time.time()
    all_m = 0
    all_f = 0
    for i in range(100):
        time_m1, time_f1 = inference()
        all_m += time_m1
        all_f += time_f1
    time2 = time.time()
    print("cost", time2-time1, (time2-time1)/100)
    print("model", all_m, "feature", all_f, "rate", all_m/all_f)
