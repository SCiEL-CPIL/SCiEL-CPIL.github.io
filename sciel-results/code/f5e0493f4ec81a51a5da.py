"""Symbolic regression: sparse RBF + polynomial basis with fixed geometry."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

# Precompute RBF centers using a Halton-like deterministic grid over [-1, 1]^2
# We use 8 RBF centers (leaving 2 params for a linear/bias term or extra basis)
# Actually, params has length 10. Let's use 7 RBFs + 3 polynomial terms (1, x, y)
# Or 8 RBFs + 2 polynomial terms (1, x*y)
# Let's stick to the directive: 10 RBFs? No, that might be overfitting with 200 points.
# Let's use 5 RBFs + 5 polynomial basis functions.
# Polynomial basis: 1, x, y, x^2, y^2, xy, x^2y, xy^2, x^3, y^3 ... too many.
# Let's use: 1, x, y, x^2+y^2, x*y (5 terms)
# RBF centers: 5 points from a Halton sequence or grid.

# Generate 5 centers deterministically
def _generate_centers(n):
    # Use a simple deterministic spread
    # Halton sequence for 2D
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    centers = []
    for i in range(1, n + 1):
        x1 = halton(i, 2) * 2.0 - 1.0
        x2 = halton(i, 3) * 2.0 - 1.0
        centers.append([x1, x2])
    return np.array(centers)

RBF_CENTERS = _generate_centers(5)
RBF_WIDTH = 0.4  # Fixed width

# Precompute the RBF matrix for a given x, y
def _compute_rbf_matrix(x, y):
    # x, y are arrays of length N
    # Returns matrix of shape (N, 5)
    N = len(x)
    M = np.zeros((N, 5))
    for j in range(5):
        cx, cy = RBF_CENTERS[j]
        dist_sq = (x - cx)**2 + (y - cy)**2
        M[:, j] = np.exp(-dist_sq / (2.0 * RBF_WIDTH**2))
    return M

# Polynomial basis: 1, x, y, x^2, y^2
def _compute_poly_matrix(x, y):
    N = len(x)
    P = np.zeros((N, 5))
    P[:, 0] = 1.0
    P[:, 1] = x
    P[:, 2] = y
    P[:, 3] = x**2
    P[:, 4] = y**2
    return P


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params: [a0..a4 (RBF amplitudes), b0..b4 (poly coeffs)]
    # Total 10 params.
    # Combine bases
    # For efficiency in the optimizer, we compute the full design matrix A (N x 10)
    # prediction = A @ params
    
    # We need to handle the case where the optimizer calls equation with the same x, y
    # but different params. The design matrix A depends only on x, y.
    # However, equation is called with params each time. We can't easily cache A unless we
    # assume x, y don't change during the fit. They don't.
    # But equation is a stateless function. We can't store state between calls easily without
    # global variables or closures, which might be risky if the evaluator resets state or
    # if multiple fits are run. 
    # Wait, the evaluator calls equation(x, y, params) repeatedly with the same x, y 
    # during the minimize step. 
    # If we recompute A every time, it's slow but safe. 
    # Let's check the cost. N=200. 5 RBFs + 5 Polys. 
    # Computing A takes ~200*10 operations. Very fast. 
    # So just recompute A every time.
    
    A_rbf = _compute_rbf_matrix(x, y)
    A_poly = _compute_poly_matrix(x, y)
    A = np.hstack((A_rbf, A_poly))
    
    # params is length 10
    # prediction = A @ params
    pred = A @ params
    
    return pred
# EVOLVE-BLOCK-END
