# ca_imaging_funhouse

Proof of concept source code for pre-processing calcuim imaging movies. 

## run_svd.py

We start with a avi movie (in the movment the parameters are set to 30fps). First -- if neccessary -- the avi file is converted into a npy file, saves on disk and then loaded again. Several steps of cleaning the signal are done. In the end we get a file that is downsampled to 3fps and reduced in x and y dimensions by a factor 2 each. A file with the ending ..._decorrelated.npy is created.

## show.py and show_b.py

Visual inspection of the movie stored in ..._decorrelated.npy. The two show files use different methods to scale the movie.

## initial_cell_estimate.py

Estimates where cells (or better areas with correlated activites are). 

- threshold: float = 0.8 # Correlation threshold. results in the area per detected cell. The smaller the value, the bigger the area.
- minimum_area: int = 100 # We don't want a huge amount of mini "cells", this we have a threshold for the area a cell needs to occupy. 

## inspection.py

Uses the estimated cell areas and trys to extract a common signal from each area (including trying to reject the non-common noise). The results are sorted by skewness. The larger the skewness, the bigger are the spike signals. 


# TODO / Known problems:

The movement jitter surpression does not work correctly. During programming it wasn't clear what could move or not in the image. Also what are stable landmarks? In the moment, the software is too much focused on the aperture. 

# Installation 

The code was tested on a Python 3.11.2 (Linux) with the following pip packages installed:

numpy scipy pandas flake8 pep8-naming black matplotlib seaborn ipython jupyterlab mypy dataclasses-json dataconf mat73 ipympl torch torchtext pywavelets scikit-image opencv-python scikit-learn tensorflow_datasets tensorboard tqdm argh sympy jsmin pybind11 pybind11-stubgen pigar asciichartpy torchvision torchaudio tensorflow natsort roipoly

Not all packages are necessary (probably these are enougth: torch torchaudio torchvision roipoly natsort numpy matplotlib) but this is our default in-house installation plus roipoly.

We used a RTX 3090 as test GPU.

For installing torch under Windows see here: https://pytorch.org/get-started/locally/
