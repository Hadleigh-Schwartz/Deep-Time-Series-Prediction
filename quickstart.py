import sys
sys.path.append("/home/hadleigh/acoustic_embedding/")

from universal_data_loader import UniversalGigaSpeechDataLoader

from deepseries.model.wave2wave import Wave2Wave
from deepseries.train import Learner
from deepseries.data import Value, create_seq2seq_data_loader, forward_split
from deepseries.nn import RMSE, MSE
import deepseries.functional as F
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

root_dir = 'my_test'

# model parameters
enc_len = 36
dec_len = 36

# here, if *_size == dec_len, we are asking the model to predict the last samples of each train_data as its test/validation. So there will only be one test/val sample per train_data.
valid_size = 36
test_size = 36 

def load_data(test_on_unseen = False):
    universal_loader = UniversalGigaSpeechDataLoader("xs", sample_rate=16000)
    train_data, valid_data, test_data = universal_loader.get_wavenet_data()

    if not test_on_unseen:
        # perform testing and validation on subsets of the train_data used for training instead of completely new train_data, as done in the original examples.
        
        train_data_series_len = train_data.shape[2]

        train_idx, valid_idx = forward_split(np.arange(train_data_series_len), enc_len=enc_len, valid_size=valid_size+test_size)
        valid_idx, test_idx = forward_split(valid_idx, enc_len, test_size)

        # Normalize the training data.  mask test so it will not be used for calculating mean/std.
        mask = np.zeros_like(train_data).astype(bool)
        mask[:, :, test_idx] = False
        train_data, mu, std = F.normalize(train_data, axis=2, fillna=True, mask=mask)

        # set up wave2wave train data in format expeceted by provided code
        train_series = train_data[:, :, train_idx]
        valid_series = train_data[:, :, valid_idx]
        test_series = train_data[:, :, test_idx]

        return train_series, valid_series, test_series
    else:
        raise NotImplementedError("Test on unseen data not implemented yet.")

def train(epochs, batch_size, lr):
    print("In train. Loading data...")
    train_data, valid_data, test_data = load_data()
    print(f"Loaded data. Training data shape: {train_data.shape}, Validation data shape: {valid_data.shape}, Test data shape: {test_data.shape}")
    train_dl = create_seq2seq_data_loader(train_data, enc_len, dec_len, sampling_rate=0.1,
                                        batch_size=batch_size, seq_last=True, device='cuda')
    valid_dl = create_seq2seq_data_loader(valid_data, enc_len, dec_len,
                                        batch_size=batch_size, seq_last=True, device='cuda')

    wave = Wave2Wave(target_size=1, num_layers=6, num_blocks=1, dropout=0.1, loss_fn=RMSE())
    wave.cuda()
    opt = torch.optim.Adam(wave.parameters(), lr=lr)
    wave_learner = Learner(wave, opt, root_dir=root_dir)
    wave_learner.fit(max_epochs=epochs, train_dl=train_dl, valid_dl=valid_dl, early_stopping=False, patient=16)

def test(load_epoch):
    # set up model
    wave = Wave2Wave(target_size=1, num_layers=6, num_blocks=1, dropout=0.1, loss_fn=RMSE())
    wave.cuda()
    opt = torch.optim.Adam(wave.parameters(), lr=0) # not used, just to satisfy Learner
    wave_learner = Learner(wave, opt, root_dir=root_dir)
    checkpoint_dir = f"{root_dir}/checkpoints"
    print(f"Loading model from epoch {load_epoch}...")
    wave_learner.load(epoch=load_epoch, checkpoint_dir=checkpoint_dir)
    print(f"Model loaded.")

    # load data
    print("Loading data...")
    _, _, test_data = load_data()
    print(f"Loaded data. Test data shape: {test_data.shape}")

    # perform inference and time it
    print("Performing inference...")
    start = time.time()
    num_series = test_data.shape[0]
    wave_preds = wave_learner.model.predict(torch.tensor(test_data[:, :, :-dec_len]).float().cuda(), dec_len).cpu().numpy().reshape(num_series, dec_len)
    wave_labels = test_data[:, :, -dec_len:]
    # TODO: compute loss
    end = time.time()
    print(f"Total inference time: {end - start} for {num_series} outputs of {dec_len} samples each.")
    for i in range(num_series):
        plt.plot(test_data[i, :, -(enc_len + dec_len):-dec_len].reshape(-1))
        plt.plot(np.arange(enc_len, enc_len + dec_len), wave_preds[i], label="wave2wave preds")
        plt.plot(np.arange(enc_len, enc_len + dec_len), test_data[i, :, -dec_len:].reshape(-1), label="target")
        plt.legend()
        plt.savefig(f"wave2wave{i}.png")    
        plt.clf()


# train(epochs=3, batch_size=16, lr=0.001)
test(load_epoch=3)
