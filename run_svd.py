import torch
import numpy as np
import os

import torchvision as tv


from svd import (
    calculate_svd,
    to_remove,
    temporal_filter,
    svd_denoise,
    convert_avi_to_npy,
    calculate_translation,
)

from ImageAlignment import ImageAlignment


if __name__ == "__main__":
    filename: str = "example_data_crop"
    window_size: int = 2
    kernel_size_pooling: int = 2
    orig_freq: int = 30
    new_freq: int = 3
    filtfilt_chuck_size: int = 10
    bp_low_frequency: float = 0.1
    bp_high_frequency: float = 1.0
    fill_value: float = 0.0
    convert_overwrite: bool | None = None

    torch_device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    np.save("kernel_size_pooling.npy", np.array(kernel_size_pooling))
    np.save("fill_value.npy", np.array(fill_value))

    if (
        (convert_overwrite is None) and (os.path.isfile(filename + ".npy") is False)
    ) or (convert_overwrite):
        print("Convert AVI file to npy file.")
        input = convert_avi_to_npy(filename)
        print("--==-- DONE --==--")
    else:
        print("Load data")
        input = np.load(filename + str(".npy"))

    with torch.no_grad():
        data = torch.tensor(input, device=torch_device)

        print("Movement compensation [BROKEN!!!!]")
        print("During development, information about what could move was missing.")
        print("Thus the preprocessing before shift determination may not work.")
        # TODO:
        data -= data.min(dim=0)[0]
        data /= data.std(dim=0, keepdim=True) + 1e-20

        image_alignment = ImageAlignment(
            default_dtype=torch.float32, device=torch_device
        )

        tvec = calculate_translation(
            input=data,
            reference_image=data[0, ...].clone(),
            image_alignment=image_alignment,
        )
        np.save(filename + "_tvec.npy", tvec.cpu().numpy())

        tvec_media = tvec.median(dim=0)[0]
        print(f"Median of movement: {tvec_media[0]}, {tvec_media[1]}")

        del data
        data = torch.tensor(input, device=torch_device)

        for id in range(0, data.shape[0]):
            data[id, ...] = tv.transforms.functional.affine(
                img=data[id, ...].unsqueeze(0),
                angle=0,
                translate=[tvec[id, 1], tvec[id, 0]],
                scale=1.0,
                shear=0,
                fill=fill_value,
            ).squeeze(0)
        data -= data.min(dim=0)[0]

        print("SVD")
        whiten_mean, whiten_k, eigenvalues = calculate_svd(data)

        np.savez(
            filename + "_svd.npz",
            whiten_mean=whiten_mean.cpu().numpy(),
            whiten_k=whiten_k.cpu().numpy(),
            eigenvalues=eigenvalues.cpu().numpy(),
        )

        print("Calculate to_remove")
        del data
        data = torch.tensor(input, device=torch_device)
        for id in range(0, data.shape[0]):
            data[id, ...] = tv.transforms.functional.affine(
                img=data[id, ...].unsqueeze(0),
                angle=0,
                translate=[tvec[id, 1], tvec[id, 0]],
                scale=1.0,
                shear=0,
                fill=fill_value,
            ).squeeze(0)
        data -= data.min(dim=0)[0]

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
