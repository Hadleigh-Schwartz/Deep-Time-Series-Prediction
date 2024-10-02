# encoding: utf-8
"""
@author : zhirui zhou
@contact: evilpsycho42@gmail.com
@time   : 2020/6/3 10:48
"""
import sys
sys.path.append("/home/hadleigh/acoustic_embedding/Deep-Time-Series-Prediction")

from deepseries.model.wave2wave import Wave2Wave
from deepseries.train import Learner
from deepseries.data import Value, create_seq2seq_data_loader, forward_split
from deepseries.nn import RMSE, MSE
import deepseries.functional as F
import numpy as np
import torch
from scipy.io import wavfile
import time

root_dir = 'wave3'

num_series = 50
batch_size = 16
enc_len = 36
dec_len = 36

epoch = 2000
lr = 0.001

# here, if *_size == dec_len, we are asking the model to predict the last samples of each series as its test/validation. So there will only be one test/val sample per series.
valid_size = 36
test_size = 36 

audio_arrays = []
min_len = float('inf')
for i in range(num_series):
    audio = wavfile.read(f"/home/hadleigh/gigaspeech_xs_samples/train_sample_{i}.wav")[1]
    if len(audio) < min_len:
        min_len = len(audio)
    audio_arrays.append(audio)
# create a numpy array of shape (N, 1, series_len), where N is the number of audio files and series_len = min_len
series = np.zeros((len(audio_arrays), 1, min_len))
for i, audio in enumerate(audio_arrays):
    series[i, 0, :] = audio[:min_len]
series_len = min_len
# series = series.reshape(1, 1, -1)
print("Series shape: ", series.shape)

train_idx, valid_idx = forward_split(np.arange(series_len), enc_len=enc_len, valid_size=valid_size+test_size)
valid_idx, test_idx = forward_split(valid_idx, enc_len, test_size)


"""
Data processing procedure:
1) Create numpy array of shape (N, 1, series_len), where N is the number of samples.
2) Pass list of indices ranging from 0 to series_len to forward_split to get train, valid, and test indices for corresponding enc_len and validation set size.
This returns the indices of each series that will be used for training and validation. For example if valid_size = 10000, then the last 10000 + enc_len samples 
of each series will be reserved for validation, and all samples up to the 1000th to last sample will be used for training.
3) Normalize the series using the normalize function from the functional module (i.e., F.normalize). This function returns the normalized series, mean, and standard deviation.
4) Pass the training portion and validation portion of the series to the create_seq2seq_data_loader function to create the training and validation data loaders.
    create_seq2seq will first convert the series into a Value object, which, assuming usage as below, simply contains the series array and additional information as well 
    as a function for returning a batch of data from the series data array, considering both the series id and desired time indexes of each series. 
    Internally, it will then create from the Value object a PyTorch-compatible dataset, which provides special feed_x and feed_y variables for the input and output of the network upon calls.
    The Pytorch data loader is then created from this dataset. Unclear how the sampling rate is used in this function other than in __len__ of sampler...

"""

# Normalize. mask test so it will not be used for calculating mean/std.
mask = np.zeros_like(series).astype(bool)
mask[:, :, test_idx] = False
series, mu, std = F.normalize(series, axis=2, fillna=True, mask=mask)

# wave2wave train
train_dl = create_seq2seq_data_loader(series[:, :, train_idx], enc_len, dec_len, sampling_rate=0.1,
                                      batch_size=batch_size, seq_last=True, device='cuda')
valid_dl = create_seq2seq_data_loader(series[:, :, valid_idx], enc_len, dec_len,
                                      batch_size=batch_size, seq_last=True, device='cuda')

wave = Wave2Wave(target_size=1, num_layers=6, num_blocks=1, dropout=0.1, loss_fn=RMSE())
wave.cuda()
opt = torch.optim.Adam(wave.parameters(), lr=lr)
wave_learner = Learner(wave, opt, root_dir=root_dir, )
wave_learner.fit(max_epochs=epoch, train_dl=train_dl, valid_dl=valid_dl, early_stopping=False, patient=16)
# wave_learner.load(epoch=3, checkpoint_dir="./wave/checkpoints")



import matplotlib.pyplot as plt

start = time.time()
wave_preds = wave_learner.model.predict(torch.tensor(series[:, :, test_idx[:-dec_len]]).float().cuda(), dec_len).cpu().numpy().reshape(num_series, dec_len)
end = time.time()
print(f"Total inference time: {end - start} for {num_series} outputs of {dec_len} samples each.")
for i in range(num_series):
    plt.plot(series[i, :, -(enc_len + dec_len):-dec_len].reshape(-1))
    plt.plot(np.arange(enc_len, enc_len + dec_len), wave_preds[i], label="wave2wave preds")
    plt.plot(np.arange(enc_len, enc_len + dec_len), series[i, :, test_idx[-dec_len:]].reshape(-1), label="target")
    plt.legend()
    plt.savefig(f"wave2wave{i}.png")    
    plt.clf()