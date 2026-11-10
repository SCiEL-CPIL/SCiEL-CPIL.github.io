"""Photonic-crystal TM band-gap design: central air hole in dielectric background."""
import time
import math
import numpy as np
from scipy.fft import fft2

_START = None
_DEADLINE = 210.0

def _now():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START

def _fast_gap(cell, eps_high, n_pw=121, n_kpts=8):
    size = cell.shape[0]
    eps = 1.0 + (eps_high - 1.0) * cell.astype(np.float64)
    inv_eps = 1.0 / eps
    fft_vals = fft2(inv_eps) / (size * size)

    N = int(math.sqrt(n_pw))
    G_coeffs = np.zeros((N, N), dtype=np.complex128)
    for gi in range(N):
        nxi = gi if gi < N // 2 else gi - N
        for gj in range(N):
            nyj = gj if gj < N // 2 else gj - N
            G_coeffs[gi, gj] = fft_vals[nxi % size, nyj % size]

    k_points = [
        (0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.5, 0.25),
        (0.5, 0.5), (0.25, 0.25), (0.125, 0.125), (0.375, 0.25),
    ]

    max_omega4 = 0.0
    min_omega5 = float('inf')

    Gx = np.array([2 * math.pi * (gi if gi < N // 2 else gi - N) for gi in range(N)])
    Gy = np.array([2 * math.pi * (gj if gj < N // 2 else gj - N) for gj in range(N)])

    for kx, ky in k_points:
        kmag2 = (kx + Gx[:, None]) ** 2 + (ky + Gy[None, :]) ** 2
        M = N * N
        A = np.zeros((M, M), dtype=np.complex128)
        for gi in range(N):
            for gj in range(N):
                row = gi * N + gj
                for gi2 in range(N):
                    for gj2 in range(N):
                        col = gi2 * N + gj2
                        A[row, col] = kmag2[gi, gj] * G_coeffs[(gi - gi2) % N, (gj - gj2) % N]

        eigvals = np.linalg.eigvalsh(A)
        if len(eigvals) >= 5:
            w4 = math.sqrt(max(eigvals[3], 0))
            w5 = math.sqrt(max(eigvals[4], 0))
            max_omega4 = max(max_omega4, w4)
            min_omega5 = min(min_omega5, w5)

    if min_omega5 == float('inf') or max_omega4 == 0:
        return -1.0
    return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)

def design(problem):
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    half = size // 2
    deadline = _DEADLINE

    cell = np.ones((size, size), dtype=np.int8)
    for i in range(size):
        for j in range(size):
            x = (i + 0.5) / size - 0.5
            y = (j + 0.5) / size - 0.5
            if x * x + y * y < 0.2 ** 2:
                cell[i, j] = 0

    best_cell = cell.copy()
    best_score = _fast_gap(cell, eps_high)

    orbits = []
    for i in range(half):
        for j in range(half):
            orbit = [(i, j), (size - 1 - i, j), (i, size - 1 - j), (size - 1 - i, size - 1 - j)]
            unique = []
            seen = set()
            for pos in orbit:
                if pos not in seen:
                    seen.add(pos)
                    unique.append(pos)
            orbits.append(unique)

    improved = True
    while improved and _now() < deadline - 20:
        improved = False
        for orbit in orbits:
            if _now() >= deadline - 20:
                break
            for pos in orbit:
                cell[pos] = 1 - cell[pos]
            new_score = _fast_gap(cell, eps_high)
            if new_score > best_score:
                best_score = new_score
                best_cell = cell.copy()
                improved = True
            else:
                for pos in orbit:
                    cell[pos] = 1 - cell[pos]

    result = [[int(best_cell[i, j]) for j in range(size)] for i in range(size)]
    return result
