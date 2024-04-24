from pathlib import Path
from svdi.svd.result import SVDResult
import numpy as np
import struct
from svdi.constant import INT_SIZE, FLOAT_SIZE, NUMBER_OF_CHANNELS


def save_as_svdi(path: Path, m, n, k, channels: tuple[SVDResult, ...]):
    with open(path, "wb") as f:
        f.write(b"SVDI")
        f.write(struct.pack("<III", m, n, k))

        for channel in channels:
            f.write(channel.U.astype(np.float32).tobytes())
            f.write(channel.S.astype(np.float32).tobytes())
            f.write(channel.Vh.astype(np.float32).tobytes())


def load_from_svdi(path: Path) -> tuple[int, int, int, tuple[SVDResult, ...]]:
    with open(path, "rb") as f:
        if f.read(4) != b"SVDI":
            raise ValueError("Not an SVDI file")

        m, n, k = struct.unpack("<III", f.read(3 * INT_SIZE))
        channels = []
        for _ in range(NUMBER_OF_CHANNELS):
            U = np.frombuffer(f.read(m * k * FLOAT_SIZE), dtype=np.float32).reshape(
                (m, k)
            )
            S = np.frombuffer(f.read(k * FLOAT_SIZE), dtype=np.float32)
            Vh = np.frombuffer(f.read(k * n * FLOAT_SIZE), dtype=np.float32).reshape(
                (k, n)
            )
            channels.append(SVDResult(U, S, Vh))

    return m, n, k, channels
