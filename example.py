import numpy as np
from utils import standarize_channels, standarize_signal
from medusa.epoching import get_epochs
import torch
#%% Now EEGOAR-Net is available both in Tensorflow and Pytorch

frameworks = ['tf','torch']
used_framework = frameworks[1] # Choose your framework

#%% Load EEG signal and EEGOAR-Net

if used_framework == 'tf':
    from model.eegoarnet_tf import EEGOARNET
    eegoarnet = EEGOARNET()
    eegoarnet.load_weights(r'weights\EEGOARNet_tf_weights.h5')
elif used_framework == 'torch':
    from model.eegoarnet_pytorch import EEGOARNET
    eegoarnet = EEGOARNET().to('cuda')
    eegoarnet.load_state_dict(torch.load(r'weights\EEGOARNet_torch_weights.pt',weights_only=True),
                              strict=True)
recordings = np.load(r'materials\eeg_examples.npy',allow_pickle=True)[()]
signal_dict = recordings['Subjects']

# Retrieve information from the recordings
fs = recordings['fs']
channels = recordings['channel_labels']

# Load 64-channel labels
sixtyfour_ch = list(np.load(r'materials\channel_set_64ch.npy',
                            allow_pickle= True)[()])
#%% We extract the EEG channel indices from our record corresponding to the
#   64-channel montage. Also, the mask corresponding to our montage.

idx_original_channels, original_masked_channels = standarize_channels(channels)

# We took the EEG signal from one of the 3 example subjects.
eeg_signal_original = signal_dict['S2']

# We adapt the signal to the EEGOAR-Net input dimensions [n_epochs, 128, 64].
# In this process we divide our signal into 1 s epochs (128 samples) and expand
# the channels from 16 to 64. For the latter we will add signals composed of
# zeros in all those channels that are not included in our original signal.
std_signal,_ = standarize_signal(eeg_signal_original[None,...],channels,
                                 idx_original_channels)
short_std_epochs = get_epochs(std_signal[0,...], 128)

#%% Apply the model. Note that we pass as second input the channel mask repeated
#   as many times as signal epochs we have.

if used_framework == 'tf':
    clean_epochs = eegoarnet.predict([short_std_epochs.copy()[:, :, :, None],
                                      np.tile(original_masked_channels[0],
                                              (
                                                  short_std_epochs.shape[
                                                      0], 1))])
elif used_framework == 'torch':
    clean_epochs = eegoarnet(
        torch.from_numpy(short_std_epochs[:, :, :, None]).float().to('cuda'),
        torch.from_numpy(np.tile(original_masked_channels[0],
                                  (short_std_epochs.shape[0], 1))).float().to('cuda')
    )
    # Convert into numpy object again
    clean_epochs = clean_epochs.cpu().detach().numpy()

# Finally, the original and cleaned signals are reshaped to recover the original
# dimensions (16 channels).
clean_reshaped = clean_epochs.reshape(
    short_std_epochs.shape[0] * short_std_epochs.shape[1],
    short_std_epochs.shape[2])[:, original_masked_channels[0]]
original_reshaped = short_std_epochs.reshape(
    short_std_epochs.shape[0] * short_std_epochs.shape[1],
    short_std_epochs.shape[2])[:, original_masked_channels[0]]
#%% OPTIONAL: Plot the signals and compare
from medusa.analysis.time_plot.time_plot import TimePlotManager, TimePlot

# We use interactive signal time plot from medusa-kernel python package.

# Define a plot manager
time_plot_manager = TimePlotManager()

# Create time plot
time_plot = TimePlot(ch_to_show=16, units="μV", initial_window_s=20)
time_plot_manager.set_time_plot(time_plot)


# Add to plot the original uncleaned signal
time_plot.add_plot(
    signal=original_reshaped,
    times=np.linspace(0, original_reshaped.shape[0] / fs, original_reshaped.shape[0]),
    start_from_zero=True,
    ch_labels=channels,
    signal_label='EEG Original',
    color='r'
)

time_plot.add_plot(
    signal=clean_reshaped,
    times=np.linspace(0, clean_reshaped.shape[0] / fs, clean_reshaped.shape[0]),
    start_from_zero=True,
    ch_labels=channels,
    signal_label='EEG Cleaned',
    color='k')

time_plot_manager.show()

