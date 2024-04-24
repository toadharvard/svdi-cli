# Image compression and decompression using SVD

## Installation
Install CLI using pipx:
```bash
pipx install git+https://github.com/toadharvard/svdi-cli.git
```
Or
```bash
pipx install svdi
```

## Usage example
```bash
svdi compress --in-file=images/rafiq.bmp --out-file=images/liquidated.svdi --compression=3 --method=numpy

svdi decompress --in-file=images/liquidated.svdi --out-file=images/rafiq2.bmp

ls -l ./images
```
## Available SVD functions
1. `rsvd` —  N Halko, P. G Martinsson, and J. A Tropp. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. Siam Review, 53(2):217-288, 2011.
2. `numpy` — NumPy's `np.linalg.svd` function.
3. `pcafast` — H. Li, G. C. Linderman, A. Szlam, K. P. Stanton, Y. Kluger, and M. Tygert. Algorithm 971: An implementation of a randomized algorithm for principal component analysis. Acm Transactions on Mathematical Software, 43(3):1-14, 2017.
4. `pi` — Power iterations method. URL: http://www.cs.yale.edu/homes/el327/datamining2013aFiles/07_singular_value_decomposition.pdf


## Available commands
```bash
svdi --help
svdi compress --help
svdi decompress --help
```

## SVDI Format

The CLI uses a simple binary format for storing SVD results in `.svdi` files.

Each .svdi file consists of a header followed by the SVD results for each channel:

1. Header
   1. Signature: A fixed 4-byte sequence b'SVDI' to identify the file format.
   2. Dimensions: Three 4-byte unsigned integers representing the shape of the matrices: m, n, and k.
       * m: The number of rows in each U matrix.
       * n: The number of columns in each Vh matrix.
       * k: The size of each S vector, and the number of columns in U and rows in Vh.
2. Channel Data
   Repeated for each channel (**NUMBER_OF_CHANNELS**):
      * U Matrix: A matrix of size m * k, with elements stored as 32-bit floats in row-major order.
      * S Vector: A vector of length k, with elements stored as 32-bit floats.
      * Vh Matrix: A matrix of size k * n, with elements stored as 32-bit floats in row-major order.

## Licence
See the details of the license in the [LICENCE](./LICENCE) file.
