import numpy as np
import torch
import matplotlib.pyplot as plt
import skimage

filename: str = "example_data_crop"
threshold: float = 0.8
tolerance: float | None = None
minimum_area: int = 100

torch_device: torch.device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print("Load data")
input = np.load(filename + str("_decorrelated.npy"))
data = torch.tensor(input, device=torch_device)
del input
print("loading done")

data = data.nan_to_num(nan=0.0)
data -= data.mean(dim=0, keepdim=True)
data /= data.std(dim=0, keepdim=True)

master_image = (data.max(dim=0)[0] - data.min(dim=0)[0]).nan_to_num(nan=0.0).clone()
temp_image = master_image.clone()
master_mask = torch.ones_like(temp_image)

stored_contours: list = []
counter: int = 0
contours_found: int = 0
while int(master_mask.sum()) > 0:
    if counter % 100 == 0:
        print(
            f"number of pixel tested: {counter} remaining pixels: {int(master_mask.sum())} cells found: {contours_found}"
        )
    counter += 1
    mask: np.ndarray | None = None

    temp_image *= master_mask

    # Convert index to 2D
    temp_idx = temp_image.argmax()
    x = int(temp_idx // int(temp_image.shape[1]))
    y = int(temp_idx - x * int(temp_image.shape[1]))
    if bool(master_mask[x, y]) is False:
        break

    test_data = data[:, x, y].clone()

    # Calculate the correlation
    scale = (data * test_data.unsqueeze(-1).unsqueeze(-1)).mean(dim=0)
    scale = scale.nan_to_num(nan=0.0)
    scale *= master_mask

    # Check for areas with high correlation
    image = (scale > threshold).type(torch.uint8).cpu().numpy()

    found_something: bool = False
    # Find the coutours
    for contour in skimage.measure.find_contours(image, 0):
        # soften outline
        if tolerance is not None:
            coords = skimage.measure.approximate_polygon(
                contour, tolerance=tolerance
            ).astype(dtype=np.float32)
        else:
            coords = contour.astype(dtype=np.float32)

        # Make a mask out of the polygon
        mask = skimage.draw.polygon2mask(scale.shape, coords)
        assert mask is not None

        # check if this is the contour in which the original point was
        if mask[x, y]:
            found_something = True

            if mask.sum() > minimum_area:
                stored_contours.append(coords)
                contours_found += 1
            idx_set_mask = torch.where(torch.tensor(mask, device=torch_device) > 0)

            master_mask[idx_set_mask] = 0.0
            break

    if found_something is False:
        master_mask[x, y] = 0.0
print("-==- DONE -==-")
np.save("cells.npy", np.array(stored_contours, dtype=object))

plt.imshow(master_image.cpu(), cmap="hot")
for i in range(0, len(stored_contours)):
    plt.plot(stored_contours[i][:, 1], stored_contours[i][:, 0], "-g", linewidth=2)
plt.colorbar()
plt.show()
