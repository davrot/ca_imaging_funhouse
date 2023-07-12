import torch
import numpy as np
from svd import calculate_svd, to_remove, temporal_filter, svd_denoise

if __name__ == "__main__":
    filename: str = "example_data_crop"
    window_size: int = 2
    kernel_size_pooling: int = 2
    orig_freq: int = 30
    new_freq: int = 3
    filtfilt_chuck_size: int = 10
    bp_low_frequency: float = 0.1
    bp_high_frequency: float = 1.0

    torch_device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print("Load data")
    input = np.load(filename + str(".npy"))
    data = torch.tensor(input, device=torch_device)

    print("Movement compensation [MISSING!!!!]")
    print("(include ImageAlignment.py into processing chain)")

    print("SVD")
    whiten_mean, whiten_k, eigenvalues = calculate_svd(data)

    print("Calculate to_remove")
    data = torch.tensor(input, device=torch_device)
    to_remove_data = to_remove(data, whiten_k, whiten_mean)

    data -= to_remove_data
    del to_remove_data

    print("apply temporal filter")
    data = temporal_filter(
        data,
        device=torch_device,
        orig_freq=orig_freq,
        new_freq=new_freq,
        filtfilt_chuck_size=filtfilt_chuck_size,
        bp_low_frequency=bp_low_frequency,
        bp_high_frequency=bp_high_frequency,
    )

    print("SVD Denosing")
    data_out = svd_denoise(data, window_size=window_size)

    print("Pooling")
    avage_pooling = torch.nn.AvgPool2d(
        kernel_size=(kernel_size_pooling, kernel_size_pooling),
        stride=(kernel_size_pooling, kernel_size_pooling),
    )
    data_out = avage_pooling(data_out)

    np.save(filename + str("_decorrelated.npy"), data_out.cpu())
