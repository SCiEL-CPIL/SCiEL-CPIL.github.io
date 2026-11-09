"""Piecewise cubic B-spline model for symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cubic_bspline_basis(x: np.ndarray) -> np.ndarray:
    """
    Compute cubic B-spline basis functions for fixed clamped knot vector.
    Returns matrix of shape (len(x), 10).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = len(x)
    
    # Fixed knot vector for 10 cubic B-spline basis functions
    # 14 knots: 3 left boundary + 8 internal + 3 right boundary
    internal_knots = np.linspace(-1.5, 1.5, 8)
    knot_vector = np.concatenate([
        np.full(3, -2.0),
        internal_knots,
        np.full(3, 2.0)
    ])
    
    n_knots = len(knot_vector)  # 14
    n_basis = n_knots - 4  # 10
    
    # Cox-de Boor recursion, vectorized over x
    # Degree 0
    N = np.zeros((n, n_knots - 1), dtype=np.float64)
    for i in range(n_knots - 1):
        N[:, i] = ((x >= knot_vector[i]) & (x < knot_vector[i + 1])).astype(np.float64)
    # Handle right boundary
    N[:, -1] = ((x >= knot_vector[-2]) & (x <= knot_vector[-1])).astype(np.float64)
    
    # Higher degrees
    for p in range(1, 4):
        N_new = np.zeros((n, n_knots - 1 - p), dtype=np.float64)
        for i in range(n_knots - 1 - p):
            d1 = knot_vector[i + p] - knot_vector[i]
            d2 = knot_vector[i + p + 1] - knot_vector[i + 1]
            if d1 > 1e-12:
                N_new[:, i] += (x - knot_vector[i]) / d1 * N[:, i]
            if d2 > 1e-12:
                N_new[:, i] += (knot_vector[i + p + 1] - x) / d2 * N[:, i + 1]
        N = N_new
    
    # Clip for numerical safety
    N = np.clip(N, 0.0, 1.0)
    
    return N


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Cubic B-spline model with 10 control points.
    y(x) = sum_{i=0}^{9} w_i * N_i(x)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    if params.size < 10:
        params = np.pad(params, (0, 10 - params.size))
    elif params.size > 10:
        params = params[:10]
    
    N = _cubic_bspline_basis(x)
    result = N @ params
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
