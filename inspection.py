import numpy as np
import torch
import matplotlib.pyplot as plt
import skimage
from scipy.stats import skew

filename: str = "example_data_crop"

torch_device: torch.device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print("Load data")
input = np.load(filename + str("_decorrelated.npy"))
data = torch.tensor(input, device=torch_device)
del input
print("loading done")

stored_contours = np.load("cells.npy", allow_pickle=True)

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

    ts = (data * mask.unsqueeze(0)).nan_to_num(nan=0.0).sum(dim=(-2, -1)) / mask.sum()
    to_plot[:, id] = ts


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
