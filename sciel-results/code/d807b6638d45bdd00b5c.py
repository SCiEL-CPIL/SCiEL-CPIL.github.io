# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _compute_area_reduced(positions, n_heights=2000):
    """Compute the area using a reduced set of heights for speed during optimization."""
    n = 64
    heights = (np.arange(n_heights) + 0.5) / n_heights
    a = (np.arange(n) + 1) / n
    
    L = positions[:, np.newaxis] + a[:, np.newaxis] * heights[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    w = (1.0 - heights) / n
    diffs = np.diff(L_sorted, axis=0)
    union_lengths = w + np.sum(np.minimum(diffs, w[np.newaxis, :]), axis=0)
    area = np.mean(union_lengths)
    return area


def _compute_area_full(positions):
    """Compute the area using the full 20000 heights as specified."""
    n = 64
    heights = (np.arange(20000) + 0.5) / 20000.0
    a = (np.arange(n) + 1) / n
    
    L = positions[:, np.newaxis] + a[:, np.newaxis] * heights[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    w = (1.0 - heights) / n
    diffs = np.diff(L_sorted, axis=0)
    union_lengths = w + np.sum(np.minimum(diffs, w[np.newaxis, :]), axis=0)
    area = np.mean(union_lengths)
    return area


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Optimize base positions for Kakeya needle triangles to minimize union area.
    Uses the Keich construction as initial guess, then refines with optimization.
    """
    n = 64
    k = int(np.log2(n))
    
    # Initial guess: Keich construction
    j = np.arange(1, k + 1)
    x0 = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    # Optimize with reduced heights
    result = minimize(
        lambda x: _compute_area_reduced(x, 2000),
        x0,
        method='Powell',
        options={'maxiter': 50, 'maxfev': 3000, 'xtol': 1e-10, 'ftol': 1e-12}
    )
    
    positions = result.x
    positions = np.where(np.isfinite(positions), positions, 0.0)
    
    return positions.astype(np.float64)


# EVOLVE-BLOCK-END
