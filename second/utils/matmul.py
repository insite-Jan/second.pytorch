#! /usr/bin/env python
from __future__ import print_function


import numpy as np

from numba import cuda, float32


bpg = 50
tpb = 32
n = bpg * tpb

@cuda.jit('(float32[:,:], float32[:,:], float32[:,:])')
def cu_square_matrix_mul(A, B, C):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y

    x = tx + bx * bw
    y = ty + by * bh

    if x >= n or y >= n:
        return

    C[y, x] = 0
    for i in range(n):
        C[y, x] += A[y, i] * B[i, x]
