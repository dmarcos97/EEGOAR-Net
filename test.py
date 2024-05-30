#%%
from EEGOARNET_architecture import EEGOARNET
import numpy as np
from EEGOARNET_utils import standarize_channels, standarize_signal
from medusa.epoching import get_epochs
from medusa.plots.timeplot import time_plot

model = EEGOARNET()
model.load_weights('EEGOAR-Net_weights.h5')

eeg_signal = np.load('eeg_examples.npy')

channels = list(np.load('channels.npy',allow_pickle= True)[()])
allowed_channels = list(np.load('channel_set_64ch.npy',allow_pickle= True)[()])

idx_original_channels, original_masked_channels = standarize_channels(channels)

clean = []
original = []
for epoch in eeg_signal:
    # Extract EEG and EOG channels
    std_signal,_ = standarize_signal(epoch[None,...],channels,idx_original_channels)
    short_std_epochs = get_epochs(std_signal[0,...], 128) * 10 ** 6


    clean_epochs = model.predict([short_std_epochs.copy()[:, :, :, None],
                                  np.tile(original_masked_channels[0],
                                                           (
                                                           short_std_epochs.shape[
                                                               0], 1))])
    original.append(short_std_epochs.reshape(
        short_std_epochs.shape[0] * short_std_epochs.shape[1],
        short_std_epochs.shape[2])[:, original_masked_channels[0]])
    clean.append(clean_epochs.reshape(
        short_std_epochs.shape[0] * short_std_epochs.shape[1],
        short_std_epochs.shape[2])[:, original_masked_channels[0]])
original = np.asarray(original)
clean = np.asarray(clean)
#%%
import matplotlib.pyplot as plt
fig, axes = plt.subplots()
time_plot(original,128,list(np.array(allowed_channels)[idx_original_channels]),
          color='r',ch_to_show=10,ch_offset=50,fig=fig,axes=axes)
time_plot(clean,128,list(np.array(allowed_channels)[idx_original_channels]),
          color='k',ch_to_show=10,ch_offset=50,fig=fig,axes=axes)
