# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_area(x, n=64, M=20000):
    """Compute the union area of the n triangles given base positions x."""
    a = (np.arange(n) + 1) / n  # a_i for i=0..n-1
    # y_m = (m + 0.5)/20000 for m=0..19999
    y = (np.arange(M) + 0.5) / M  # shape (M,)
    
    # L_i = x_i + a_i * y, shape (n, M)
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1 - y[None, :]) / n  # shape (1, M)
    
    # Sort L along axis 0 for each column (height)
    L_sorted = np.sort(L, axis=0)  # shape (n, M)
    
    # Union length at each height: w + sum over consecutive pairs of min(L_{j+1} - L_j, w)
    diffs = L_sorted[1:, :] - L_sorted[:-1, :]  # shape (n-1, M)
    union_lengths = w[0] + np.sum(np.minimum(diffs, w[0]), axis=0)  # shape (M,)
    
    area = np.mean(union_lengths)
    return area


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem. For a fixed n, choose
    base positions x = (x_0, ..., x_{n-1}). For position index i (0-based) let
    a_i = (i + 1) / n and define the triangle R_i with vertices
        (x_i, 0), (x_i + 1/n, 0), (x_i + a_i, 1).
    Every triangle has base width 1/n on the line y = 0 and its apex on the line
    y = 1. The objective is to choose x so that the AREA of the union of the n
    triangles R_i is as small as possible. Here n = 64.

    This baseline uses the classical Keich construction (valid for n a power of 2):
    expand a_i = i/n in binary as a_i = sum_j eps_j 2^{-j} and set
        x_i = sum_{j=1}^{k} ((1 - j) / k) * eps_j * 2^{-j},   k = log2(n).

    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x = np.array(positions, dtype=np.float64)
    
    # Coordinate descent to improve the Keich baseline
    start_time = time.time()
    deadline = start_time + 600  # 600 second limit
    
    # Precompute constants
    a = (np.arange(n) + 1) / n  # shape (n,)
    M = 20000
    y = (np.arange(M) + 0.5) / M  # shape (M,)
    w = (1 - y) / n  # shape (M,)
    
    current_area = _compute_area(x, n, M)
    best_x = x.copy()
    best_area = current_area
    
    # Coordinate descent with decreasing step sizes
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
    
    for step in step_sizes:
        if time.time() > deadline - 30:
            break
        
        improved = True
        while improved and time.time() < deadline - 30:
            improved = False
            for i in range(n):
                if time.time() > deadline - 30:
                    break
                
                # Try small perturbations around current x[i]
                best_local = x[i]
                best_local_area = current_area
                
                # Try a few candidate offsets
                for delta in [-step, -step*0.5, step*0.5, step]:
                    x_trial = x.copy()
                    x_trial[i] = x[i] + delta
                    trial_area = _compute_area(x_trial, n, M)
                    if trial_area < best_local_area:
                        best_local_area = trial_area
                        best_local = x_trial[i]
                
                if best_local != x[i]:
                    x[i] = best_local
                    current_area = best_local_area
                    improved = True
                    if current_area < best_area:
                        best_area = current_area
                        best_x = x.copy()
    
    return best_x


# EVOLVE-BLOCK-END
