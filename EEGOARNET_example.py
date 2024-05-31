from EEGOARNET_architecture import EEGOARNET
import numpy as np
from EEGOARNET_utils import standarize_channels, standarize_signal
from medusa.epoching import get_epochs
#%% Load EEG signal and EEGOAR-Net
eegoarnet = EEGOARNET()
eegoarnet.load_weights('EEGOAR-Net_weights.h5')
recordings = np.load('materials\eeg_examples.npy',allow_pickle=True)[()]
signal_dict = recordings['Subjects']

# Retrieve information from the recordings
fs = recordings['fs']
channels = recordings['channel_labels']

# Load 64-channel labels
sixtyfour_ch = list(np.load('materials\channel_set_64ch.npy',
                            allow_pickle= True)[()])
#%% We extract the EEG channel indices from our record corresponding to the
#   64-channel montage. Also, the mask corresponding to our montage.

idx_original_channels, original_masked_channels = standarize_channels(channels)

# We took the EEG signal from one of the 3 example subjects.
eeg_signal_original = signal_dict['S1']

# We adapt the signal to the EEGOAR-Net input dimensions [n_epochs, 128, 64].
# In this process we divide our signal into 1 s epochs (128 samples) and expand
# the channels from 16 to 64. For the latter we will add signals composed of
# zeros in all those channels that are not included in our original signal.
std_signal,_ = standarize_signal(eeg_signal_original[None,...],channels,
                                 idx_original_channels)
short_std_epochs = get_epochs(std_signal[0,...], 128)

#%% Apply the model. Note that we pass as second input the channel mask repeated
#   as many times as signal epochs we have.

clean_epochs = eegoarnet.predict([short_std_epochs.copy()[:, :, :, None],
                              np.tile(original_masked_channels[0],
                                      (
                                          short_std_epochs.shape[
                                              0], 1))])

# Finally, the original and cleaned signals are reshaped to recover the original
# dimensions (16 channels).
clean_reshaped = clean_epochs.reshape(
    short_std_epochs.shape[0] * short_std_epochs.shape[1],
    short_std_epochs.shape[2])[:, original_masked_channels[0]]
original_reshaped = short_std_epochs.reshape(
    short_std_epochs.shape[0] * short_std_epochs.shape[1],
    short_std_epochs.shape[2])[:, original_masked_channels[0]]
#%% OPTIONAL: Plot the signals and compare
from medusa.plots.timeplot import time_plot
import matplotlib.pyplot as plt

# We use interactive signal time plot from medusa-kernel python package. Note
# that you should prepare your enviroment to use an interactive matplotlib
# backend.

# Define a figure and axes for the plot
fig, axes = plt.subplots()

# Set an offset to apply to the signal (the time_plot function calculates
# automatically from the signal values and, as the cleaned signal does not contain
# eye blinks, the automatic offset would be too low compared to the other signal
# offset).
ch_offset = 60


time_plot(signal=original_reshaped,fs=fs,ch_labels=channels,
          color='r',ch_to_show=10,ch_offset=60,fig=fig,axes=axes)
time_plot(clean_reshaped,128,channels,
          color='k',ch_to_show=10,ch_offset=60,fig=fig,axes=axes)
