from enum import StrEnum, auto
import os
import typer
from pathlib import Path
from PIL import Image

from svdi.svd import basic_rSVD, numpy_default, pcafast
from svdi.svd import power_iterations as pi
from .main import compress_image, save_image, decompress_image
from .file.format import save_as_svdi, load_from_svdi

app = typer.Typer(
    no_args_is_help=True,
    help="Image compression and decompression using SVD",
    pretty_exceptions_show_locals=False,
)


class Method(StrEnum):
    numpy = auto()
    pi = auto()
    rSVD = auto()
    pcafast = auto()


@app.command(no_args_is_help=True)
def compress(
    method: Method = typer.Option(Method.numpy, help="Compression method"),
    compression: float = typer.Option(2, min=1, help="Compression factor"),
    in_file: Path = typer.Option(..., help="Path to image file"),
    out_file: Path = typer.Option(..., help="Path to .svdi file to be created"),
    power_iterations: int = typer.Option(
        100,
        min=0,
        help="Number of power iterations. Applicable only to rSVD and pcafast methods",
    ),
):
    """
    Compress image to .svdi file format
    """
    in_image = Image.open(in_file)

    match method:
        case Method.numpy:
            svd_method = numpy_default.get_svd
        case Method.pi:
            svd_method = pi.get_svd
        case Method.rSVD:
            svd_method = lambda A, k: basic_rSVD.get_svd(A, k, power_iterations)
        case Method.pcafast:
            svd_method = lambda A, k: pcafast.get_svd(A, k, power_iterations)

    raw_size = os.path.getsize(in_file)
    channels = compress_image(raw_size, svd_method, in_image, compression)
    channel = channels[0]
    m = channel.U.shape[0]
    n = channel.Vh.shape[1]
    k = channel.S.shape[0]
    save_as_svdi(out_file, m, n, k, channels)


@app.command(no_args_is_help=True)
def decompress(
    in_file: Path = typer.Option(..., help="Path to .svdi file "),
    out_file: Path = typer.Option(..., help="Path to image file to be created"),
):
    """
    Decompress from .svdi file format to image
    """
    *_, compressed = load_from_svdi(in_file)
    image = decompress_image(compressed)
    save_image(out_file, image)
