from pathlib import Path
from typing import Callable
from PIL import Image
import math
import numpy as np
from svdi.svd.result import SVDResult
from svdi.constant import INT_SIZE, FLOAT_SIZE, NUMBER_OF_CHANNELS, MODE
from rich.progress import track


def save_image(out_file: Path, image: Image.Image):
    FORMAT = "PNG"
    try:
        image.save(out_file)
    except ValueError:
        image.save(out_file, format=FORMAT)


def calculate_k_value(raw_size: int, compression: float, m: int, n: int):
    new_size = raw_size / compression
    k = (new_size - 3 * INT_SIZE) / (NUMBER_OF_CHANNELS * FLOAT_SIZE * (n + 1 + m))
    if k < 1:
        raise ValueError(
            "Compression factor too high, maximum is",
            raw_size / (NUMBER_OF_CHANNELS * FLOAT_SIZE * (n + 1 + m) + 3 * INT_SIZE),
        )
    return math.floor(k)


def svd_to_matrix(svd: SVDResult) -> np.ndarray:
    return svd.U @ np.diag(svd.S) @ svd.Vh


def compress_image(
    raw_size: int,
    svd_method: Callable[[np.ndarray, int], SVDResult],
    image: Image.Image,
    compression: float,
) -> tuple[SVDResult, ...]:
    in_channels = image.convert(MODE).split()
    out_channels = []

    for channel in track(
        in_channels, description="Compressing...", total=NUMBER_OF_CHANNELS
    ):
        matrix = np.array(channel)
        m, n = matrix.shape
        k = calculate_k_value(raw_size, compression, m, n)
        svd = svd_method(matrix, k)
        out_channels.append(svd)

    return out_channels


def decompress_image(
    compressed: tuple[SVDResult, ...],
) -> Image.Image:
    out_channels = [
        svd_to_matrix(svd)
        for svd in track(
            compressed, description="Decompressing...", total=NUMBER_OF_CHANNELS
        )
    ]
    merged = np.dstack(out_channels).astype(np.uint8)
    return Image.fromarray(merged, mode=MODE)
