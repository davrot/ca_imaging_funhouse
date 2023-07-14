import numpy as np
import torch
import matplotlib.pyplot as plt
import skimage
from scipy.stats import skew

filename: str = "example_data_crop"
use_svd: bool = True
show_movie: bool = True

from Anime import Anime

torch_device: torch.device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print("Load data")
input = np.load(filename + str("_decorrelated.npy"))
data = torch.tensor(input, device=torch_device)
del input
print("loading done")

stored_contours = np.load("cells.npy", allow_pickle=True)

if use_svd:
    data_flat = torch.flatten(
        data.nan_to_num(nan=0.0).movedim(0, -1),
        start_dim=0,
        end_dim=1,
    )

to_plot = torch.zeros(
    (int(data.shape[0]), int(stored_contours.shape[0])),
    device=torch_device,
    dtype=torch.float32,
)
for id in range(0, stored_contours.shape[0]):
    mask = torch.tensor(
        skimage.draw.polygon2mask(
            (int(data.shape[1]), int(data.shape[2])), stored_contours[id]
        ),
        device=torch_device,
        dtype=torch.float32,
    )
    if use_svd:
        mask_flat = torch.flatten(
            mask.unsqueeze(0).nan_to_num(nan=0.0).movedim(0, -1),
            start_dim=0,
            end_dim=1,
        )
        idx = torch.where(mask_flat > 0)[0]
        temp = data_flat[idx, :].clone()
        whiten_mean = torch.mean(temp, dim=-1)
        temp -= whiten_mean.unsqueeze(-1)
        svd_u, svd_s, _ = torch.svd_lowrank(temp, q=6)

        whiten_k = (
            torch.sign(svd_u[0, :]).unsqueeze(0) * svd_u / (svd_s.unsqueeze(0) + 1e-20)
        )[:, 0]

        temp = temp * whiten_k.unsqueeze(-1)
        data_svd = temp.movedim(-1, 0).sum(dim=-1)
        to_plot[:, id] = data_svd
    else:
        ts = (data * mask.unsqueeze(0)).nan_to_num(nan=0.0).sum(
            dim=(-2, -1)
        ) / mask.sum()
        to_plot[:, id] = ts

with torch.no_grad():
    if show_movie:
        print("Calculate movie")
        # Clean tensor
        data *= 0.0
        for id in range(0, stored_contours.shape[0]):
            mask = torch.tensor(
                skimage.draw.polygon2mask(
                    (int(data.shape[1]), int(data.shape[2])), stored_contours[id]
                ),
                device=torch_device,
                dtype=torch.float32,
            )
            # * 1.0 - mask: otherwise the overlapping outlines look bad
            # Yes... reshape and indices would be faster...
            data *= 1.0 - mask.unsqueeze(0)
            data += mask.unsqueeze(0) * to_plot[:, id].unsqueeze(1).unsqueeze(2)

        ani = Anime()
        ani.show(data)

skew_value = skew(to_plot.cpu().numpy(), axis=0)
skew_idx = np.flip(skew_value.argsort())
skew_value = skew_value[skew_idx]

to_plot_np = to_plot.cpu().numpy()
to_plot_np = to_plot_np[:, skew_idx]

plt.plot(to_plot[:, 0:5].cpu())
plt.show()

block_size: int = 8
# print(to_plot.shape[1] // block_size)
for i in range(0, 4 * 8):
    plt.subplot(8, 4, i + 1)
    plt.plot(to_plot[:, i * block_size : (i + 1) * block_size].cpu())
plt.show()
