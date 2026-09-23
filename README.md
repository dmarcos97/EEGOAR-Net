# EEGOAR-Net

Open implementation of **EEGOAR-Net**, a deep learning model for reducing ocular artifacts in EEG signals.

This repository contains the code associated with the work:

Calibration-Free Ocular Artifact Reduction in EEG signals using a Montage-Independent Deep Learning Model.

EEGOAR-Net is designed to attenuate eye-related artifacts (such as blinks or eye movements) in EEG recordings while preserving neural information, enabling easier use in EEG research and brain–computer interface applications.

This repository includes both a **native PyTorch** (`model\eegoarnet_pytorch.py`) and **native TensorFlow/Keras** implementation (`model\eegoarnet_tf.py`).
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

├── model/

│ ├── eegoarnet_tf.py # Model architecture (TensorFlow/Keras)

│ ├── eegoarnet_torch.py # Model architecture (PyTorch)

│ ├── EEGOARNET_tf_summary.txt # TensorFlow model summary

│ └── EEGOARNET_torch.txt # PyTorch model summary

│

├── weights/

│ ├── EEGOARNet_tf_weights.h5 # Pretrained weights (TensorFlow/Keras)

│ └── EEGOARNet_torch_weights.pt # Pretrained weights (PyTorch)

│

├── materials/ # Additional materials

│

├── example.py # Example usage (TensorFlow and PyTorch)

├── EEGOAR-Net scheme.png

├── LICENSE

├── requirements_tf.txt # TensorFlow/Keras dependencies

├── requirements_torch.txt # PyTorch dependencies

└── README.md

---

# Requirements
⚠️ Important

The TensorFlow and PyTorch implementations have different requirements and
should be installed in **separate environments**.

### TensorFlow

This implementation currently works only with:

**Python <= 3.10**

Some dependencies used in this repository are not compatible with newer Python versions.

Install dependencies with:

```bash
pip install -r requirements_tf.txt
```

### PyTorch

No Python version ceiling — validated on Python 3.12. Validated with
`torch==2.5.1+cu121`; newer torch/CUDA versions are expected to work fine
(only standard ops are used: `Conv2d`, `BatchNorm2d`, `MaxPool2d`, `Upsample`,
`ELU`).

Install dependencies with:

```bash
pip install -r requirements_torch.txt
```

For a CUDA build, install `torch` from PyTorch's own index first:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

---

# Usage

`example.py` shows how to load and run both implementations.

---

# Citation 
If you use this repository in your research, please cite the associated publication: 
```bash
Marcos-Martínez, D., et al. Calibration-Free Ocular Artifact Reduction in EEG signals using a Montage-Independent Deep Learning Model.
Biomedical Signal Processing and Control, 2025. DOI: https://doi.org/10.1016/j.bspc.2025.108147

````
