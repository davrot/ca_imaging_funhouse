import numpy as np
import torch
import matplotlib.pyplot as plt
import skimage
from scipy.stats import skew
from svd import to_remove
import torchvision as tv

from ImageAlignment import ImageAlignment

from Anime import Anime

filename: str = "example_data_crop"
use_svd: bool = True
show_movie: bool = True

torch_device: torch.device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

with torch.no_grad():
    print("Load data")
    input = np.load(filename + str(".npy"))  # str("_decorrelated.npy"))
    # del input
    print("loading done")

    stored_contours = np.load("cells.npy", allow_pickle=True)
    kernel_size_pooling = int(np.load("kernel_size_pooling.npy"))
    fill_value = float(np.load("fill_value.npy"))

    image_alignment = ImageAlignment(default_dtype=torch.float32, device=torch_device)

    tvec = torch.tensor(np.load(filename + "_tvec.npy"), device=torch_device)

    np_svd_data = np.load(
        filename + "_svd.npz",
    )
    whiten_mean = torch.tensor(np_svd_data["whiten_mean"], device=torch_device)
    whiten_k = torch.tensor(np_svd_data["whiten_k"], device=torch_device)
    eigenvalues = torch.tensor(np_svd_data["eigenvalues"], device=torch_device)
    del np_svd_data

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
    data -= data.min(dim=0, keepdim=True)[0]

    to_remove_data = to_remove(data, whiten_k, whiten_mean)

    data -= to_remove_data
    del to_remove_data

    print("Pooling")
    # Warning: The contour masks have the same size as the binned data!!!
    avage_pooling = torch.nn.AvgPool2d(
        kernel_size=(kernel_size_pooling, kernel_size_pooling),
        stride=(kernel_size_pooling, kernel_size_pooling),
    )
    data = avage_pooling(data)

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

    print("Calculate cell's time series")

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
                torch.sign(svd_u[0, :]).unsqueeze(0)
                * svd_u
                / (svd_s.unsqueeze(0) + 1e-20)
            )[:, 0]

            temp = temp * whiten_k.unsqueeze(-1)
            data_svd = temp.movedim(-1, 0).sum(dim=-1)
            to_plot[:, id] = data_svd
        else:
            ts = (data * mask.unsqueeze(0)).nan_to_num(nan=0.0).sum(
                dim=(-2, -1)
            ) / mask.sum()
            to_plot[:, id] = ts

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
exit()

skew_value = skew(to_plot.cpu().numpy(), axis=0)
skew_idx = np.flip(skew_value.argsort())
skew_value = skew_value[skew_idx]

to_plot_np = to_plot.cpu().numpy()
to_plot_np = to_plot_np[:, skew_idx]


plt.imshow(to_plot_np.T, cmap="gray_r", interpolation="nearest")
plt.colorbar()
plt.show()


# plt.plot(to_plot[:, 0:5].cpu())
# plt.show()

# block_size: int = 8
# # print(to_plot.shape[1] // block_size)
# for i in range(0, 4 * 8):
#     plt.subplot(8, 4, i + 1)
#     plt.plot(to_plot[:, i * block_size : (i + 1) * block_size].cpu())
#     plt.ylim(
#         [
#             to_plot.min().cpu(),
#             to_plot.max().cpu(),
#         ]
#     )
# plt.show()
