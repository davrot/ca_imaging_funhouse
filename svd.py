import torch
import torchaudio as ta
import cv2
import numpy as np
from tqdm import trange


def convert_avi_to_npy(filename: str) -> np.ndarray:
    capture_from_file = cv2.VideoCapture(filename + str(".avi"))
    avi_length = int(capture_from_file.get(cv2.CAP_PROP_FRAME_COUNT))

    # To torch and beyond
    data: np.ndarray | None = None
    for i in trange(0, avi_length):
        read_ok, frame = capture_from_file.read()

        assert read_ok

        if data is None:
            data = np.empty(
                (avi_length, frame.shape[0], frame.shape[1]),
                dtype=np.float32,
            )
        assert data is not None
        data[i, :, :] = frame.mean(axis=-1).astype(np.float32)
    assert data is not None
    np.save(filename + str(".npy"), data)

    return data


@torch.no_grad()
def to_remove(
    data: torch.Tensor, whiten_k: torch.Tensor, whiten_mean: torch.Tensor
) -> torch.Tensor:
    whiten_k = whiten_k[:, :, 0]

    data = (data - whiten_mean.unsqueeze(0)) * whiten_k.unsqueeze(0)
    data_svd = data.sum(dim=-1).sum(dim=-1).unsqueeze(-1).unsqueeze(-1)

    factor = (data * data_svd).sum(dim=0, keepdim=True) / (data_svd**2).sum(
        dim=0, keepdim=True
    )
    to_remove = data_svd * factor
    to_remove /= whiten_k.unsqueeze(0) + 1e-20
    to_remove += whiten_mean.unsqueeze(0)

    return to_remove


@torch.no_grad()
def calculate_svd(
    input: torch.Tensor, lowrank_q: int = 6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selection = torch.flatten(
        input.clone().movedim(0, -1),
        start_dim=0,
        end_dim=1,
    )

    whiten_mean = torch.mean(selection, dim=-1)
    selection -= whiten_mean.unsqueeze(-1)
    whiten_mean = whiten_mean.reshape((input.shape[1], input.shape[2]))

    svd_u, svd_s, _ = torch.svd_lowrank(selection, q=lowrank_q)

    whiten_k = (
        torch.sign(svd_u[0, :]).unsqueeze(0) * svd_u / (svd_s.unsqueeze(0) + 1e-20)
    )
    whiten_k = whiten_k.reshape((input.shape[1], input.shape[2], svd_s.shape[0]))
    eigenvalues = svd_s

    return whiten_mean, whiten_k, eigenvalues


@torch.no_grad()
def filtfilt(
    input: torch.Tensor,
    butter_a: torch.Tensor,
    butter_b: torch.Tensor,
) -> torch.Tensor:
    assert butter_a.ndim == 1
    assert butter_b.ndim == 1
    assert butter_a.shape[0] == butter_b.shape[0]

    process_data: torch.Tensor = input.movedim(0, -1).detach().clone()

    padding_length = 12 * int(butter_a.shape[0])
    left_padding = 2 * process_data[..., 0].unsqueeze(-1) - process_data[
        ..., 1 : padding_length + 1
    ].flip(-1)
    right_padding = 2 * process_data[..., -1].unsqueeze(-1) - process_data[
        ..., -(padding_length + 1) : -1
    ].flip(-1)
    process_data_padded = torch.cat((left_padding, process_data, right_padding), dim=-1)

    output = ta.functional.filtfilt(
        process_data_padded.unsqueeze(0), butter_a, butter_b, clamp=False
    ).squeeze(0)

    output = output[..., padding_length:-padding_length].movedim(-1, 0)
    return output


@torch.no_grad()
def butter_bandpass(
    device: torch.device,
    low_frequency: float = 0.1,
    high_frequency: float = 1.0,
    fs: float = 30.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    import scipy

    butter_b_np, butter_a_np = scipy.signal.butter(
        4, [low_frequency, high_frequency], btype="bandpass", output="ba", fs=fs
    )
    butter_a = torch.tensor(butter_a_np, device=device, dtype=torch.float32)
    butter_b = torch.tensor(butter_b_np, device=device, dtype=torch.float32)
    return butter_a, butter_b


@torch.no_grad()
def chunk_iterator(array: torch.Tensor, chunk_size: int):
    for i in range(0, array.shape[0], chunk_size):
        yield array[i : i + chunk_size]


@torch.no_grad()
def bandpass(
    data: torch.Tensor,
    device: torch.device,
    low_frequency: float = 0.1,
    high_frequency: float = 1.0,
    fs=30.0,
    filtfilt_chuck_size: int = 10,
) -> torch.Tensor:
    butter_a, butter_b = butter_bandpass(
        device=device,
        low_frequency=low_frequency,
        high_frequency=high_frequency,
        fs=fs,
    )

    index_full_dataset: torch.Tensor = torch.arange(
        0, data.shape[1], device=device, dtype=torch.int64
    )

    for chunk in chunk_iterator(index_full_dataset, filtfilt_chuck_size):
        temp_filtfilt = filtfilt(
            data[:, chunk, :],
            butter_a=butter_a,
            butter_b=butter_b,
        )
        data[:, chunk, :] = temp_filtfilt

    return data


@torch.no_grad()
def temporal_filter(
    data: torch.Tensor,
    device: torch.device,
    orig_freq: int = 30,
    new_freq: int = 3,
    filtfilt_chuck_size: int = 10,
    bp_low_frequency: float = 0.1,
    bp_high_frequency: float = 1.0,
) -> torch.Tensor:
    data = ta.functional.resample(
        data.movedim(0, -1), orig_freq=orig_freq, new_freq=new_freq
    ).movedim(-1, 0)

    data = bandpass(
        data,
        device=device,
        low_frequency=bp_low_frequency,
        high_frequency=bp_high_frequency,
        fs=float(new_freq),
        filtfilt_chuck_size=filtfilt_chuck_size,
    )

    return data


@torch.no_grad()
def svd_denoise(data: torch.Tensor, window_size: int) -> torch.Tensor:
    data_out = torch.zeros_like(data)

    for x in trange(0, data.shape[1]):
        for y in range(0, data.shape[2]):
            if (
                ((x - window_size) > 0)
                and ((y - window_size) > 0)
                and ((x + window_size) <= data.shape[1])
                and ((y + window_size) <= data.shape[2])
            ):
                data_sel: torch.Tensor = data[
                    :,
                    x - window_size : x + window_size + 1,
                    y - window_size : y + window_size + 1,
                ]

                whiten_mean, whiten_k, eigenvalues = calculate_svd(data_sel.clone())
                to_remove_data = to_remove(data_sel, whiten_k, whiten_mean)
                data_out[:, x, y] = to_remove_data[:, window_size, window_size]
    return data_out


@torch.no_grad()
def calculate_translation(
    input: torch.Tensor,
    reference_image: torch.Tensor,
    image_alignment,
    start_position_coefficients: int = 0,
    batch_size: int = 100,
) -> torch.Tensor:
    tvec = torch.zeros((input.shape[0], 2))

    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(input[start_position_coefficients:, ...]),
        batch_size=batch_size,
        shuffle=False,
    )
    start_position: int = 0
    for input_batch in data_loader:
        assert len(input_batch) == 1

        end_position = start_position + input_batch[0].shape[0]

        tvec_temp = image_alignment.dry_run_translation(
            input=input_batch[0],
            new_reference_image=reference_image,
        )

        assert tvec_temp is not None

        tvec[start_position:end_position, :] = tvec_temp

        start_position += input_batch[0].shape[0]

    tvec = torch.round(tvec)
    return tvec
