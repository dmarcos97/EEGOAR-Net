# EEGOAR-Net

Open implementation of **EEGOAR-Net**, a deep learning model for reducing ocular artifacts in EEG signals.

This repository contains the code associated with the work:

Calibration-Free Ocular Artifact Reduction in EEG signals using a Montage-Independent Deep Learning Model.

EEGOAR-Net is designed to attenuate eye-related artifacts (such as blinks or eye movements) in EEG recordings while preserving neural information, enabling easier use in EEG research and brain–computer interface applications.

---

# Overview

Ocular artifacts are one of the main sources of noise in EEG signals. Traditional solutions often require calibration procedures and additional EOG channels.

EEGOAR-Net proposes a deep learning approach that:

- Reduces ocular artifacts in EEG signals
- Works across different EEG montages
- Does not require subject-specific calibration
- Preserves relevant neural information

The architecture follows an encoder-decoder style network trained to reconstruct EEG signals with reduced ocular artifact influence.

---

# Repository Structure

├── EEGOARNET_architecture.py # Model architecture

├── EEGOARNET_utils.py # Utility functions

├── EEGOARNET_example.py # Example usage

├── EEGOAR-Net_weights.h5 # Pretrained weights

├── materials/ # Additional materials

├── requirements.txt

└── README.md

# Requirements

⚠️ Important

This project currently works only with:

Python <= 3.10

Some dependencies used in this repository are not compatible with newer Python versions.

Install dependencies with:

```bash
pip install -r requirements.txt

````

# Citation 
If you use this repository in your research, please cite the associated publication: 
```bash
Marcos-Martínez, D., et al. Calibration-Free Ocular Artifact Reduction in EEG signals using a Montage-Independent Deep Learning Model.
Biomedical Signal Processing and Control, 2025. DOI: https://doi.org/10.1016/j.bspc.2025.108147

````
