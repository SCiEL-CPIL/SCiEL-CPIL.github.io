"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    n_points = len(x)
    if n_points == 0:
        return np.array([])
    
    # Piecewise linear hat function basis with 10 basis functions
    # Centers evenly spaced in [-2, 2]
    n_basis = 10
    centers = np.linspace(-2.0, 2.0, n_basis)
    spacing = centers[1] - centers[0]
    
    # Build basis matrix: each row corresponds to a point in x, each column to a hat function
    basis = np.zeros((n_points, n_basis))
    
    for i in range(n_basis):
        c = centers[i]
        # Hat function: triangular, peak at c, zero at c ± spacing
        # For boundary hats, we extend to domain edges
        if i == 0:
            # Left boundary: linear from -2 to c+spacing, zero beyond
            left = -2.0
            right = c + spacing
            # f(x) = (x - left) / (right - left) for x in [left, right], 0 otherwise
            mask = (x >= left) & (x <= right)
            basis[mask, i] = (x[mask] - left) / (right - left)
        elif i == n_basis - 1:
            # Right boundary: linear from c-spacing to 2, zero before
            left = c - spacing
            right = 2.0
            mask = (x >= left) & (x <= right)
            basis[mask, i] = (right - x[mask]) / (right - left)
        else:
            # Interior hat: triangle from c-spacing to c+spacing
            left = c - spacing
            right = c + spacing
            mid = c
            # Rising edge: [left, mid]
            mask_left = (x >= left) & (x <= mid)
            basis[mask_left, i] = (x[mask_left] - left) / (mid - left)
            # Falling edge: [mid, right]
            mask_right = (x >= mid) & (x <= right)
            basis[mask_right, i] = (right - x[mask_right]) / (right - mid)
    
    # Linear combination - strictly linear in params, so BFGS finds global LS solution
    return basis @ params
# EVOLVE-BLOCK-END
