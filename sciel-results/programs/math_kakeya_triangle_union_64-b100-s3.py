# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _compute_area(positions: np.ndarray) -> float:
    """
    Compute the area of the union of the n triangles using the specified quadrature.
    n = 64, heights y_m = (m + 0.5) / 20000 for m = 0..19999.
    """
    n = len(positions)
    num_heights = 20000
    y = (np.arange(num_heights) + 0.5) / num_heights  # shape (20000,)
    
    # a_i = (i+1)/n for i=0..n-1
    a = (np.arange(n) + 1.0) / n  # shape (n,)
    
    # For each height y, L_i = x_i + a_i * y, w = (1 - y) / n
    # L has shape (num_heights, n)
    L = positions[None, :] + a[None, :] * y[:, None]  # shape (20000, 64)
    w = (1.0 - y) / n  # shape (20000,)
    
    # Sort L along axis=1
    L_sorted = np.sort(L, axis=1)  # shape (20000, 64)
    
    # Union length at each height: w + sum_{j=0}^{n-2} min(L_{j+1} - L_j, w)
    diffs = L_sorted[:, 1:] - L_sorted[:, :-1]  # shape (20000, 63)
    min_diffs = np.minimum(diffs, w[:, None])  # shape (20000, 63)
    
    union_lengths = w + np.sum(min_diffs, axis=1)  # shape (20000,)
    
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

    This implementation uses a direct numerical optimization approach starting
    from the Keich construction as an initial guess, then refining with L-BFGS-B.

    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    k = int(np.log2(n))
    
    # Start with Keich construction as initial guess
    j = np.arange(1, k + 1)
    positions = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    # Refine with L-BFGS-B
    result = minimize(
        _compute_area,
        positions,
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    if result.success or result.fun < _compute_area(positions):
        positions = result.x
    else:
        # Keep the original if optimization didn't improve
        pass
    
    return positions.astype(np.float64)


# EVOLVE-BLOCK-END
