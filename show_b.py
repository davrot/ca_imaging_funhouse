import numpy as np
import torch
from Anime import Anime

# Convert from avi to npy
filename: str = "example_data_crop"


torch_device: torch.device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print("Load data")
input = np.load(filename + str("_decorrelated.npy"))
data = torch.tensor(input, device=torch_device)
del input
print("loading done")

data = data.nan_to_num(nan=0.0)
data -= data.min(dim=0, keepdim=True)[0]
data *= data.std(dim=0, keepdim=True)

ani = Anime()
ani.show(data)
