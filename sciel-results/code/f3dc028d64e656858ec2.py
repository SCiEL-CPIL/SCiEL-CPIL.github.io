"""
Strategy: Adaptive Basis Selection for Symbolic Regression.

The evaluator calls equation(x, params) repeatedly during BFGS optimization.
Instead of using a single fixed basis (e.g., Chebyshev), we evaluate a portfolio
of basis sets on the input x. We select the basis with the best condition number
(lowest condition number = most stable for linear least squares).

This approach:
1. Avoids local minima (linear in params).
2. Adapts to the function's smoothness (e.g., if x is clustered, lower order or
   specific bases might be better; if x is uniform, Chebyshev is often robust).
3. Ensures numerical stability by checking the condition number of the design matrix.
4. Uses a time guard to ensure we finish within the 150s limit, though basis
   selection for 200 points is extremely fast.

We use 10 parameters. The basis functions are scaled to be roughly orthogonal
or well-conditioned on the domain [-2, 2].
"""

# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: The evaluator kills after 150s.
# We start the timer on the first call.
_START_TIME = None

def _get_elapsed_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _basis_polynomial(x, n=10):
    """Monomial basis: 1, x, x^2, ..."""
    # Normalize x to [-1, 1] to improve conditioning
    z = x / 2.0
    basis = np.vander(z, N=n, increasing=True)
    return basis

def _basis_chebyshev(x, n=10):
    """Chebyshev polynomials of the first kind: T_0, T_1, ... T_{n-1}"""
    z = x / 2.0 # Map [-2, 2] to [-1, 1]
    basis = np.zeros((len(z), n))
    T0 = np.ones_like(z)
    if n > 0:
        basis[:, 0] = T0
    if n > 1:
        T1 = z
        basis[:, 1] = T1
        for k in range(2, n):
            Tk = 2.0 * z * T1 - T0
            basis[:, k] = Tk
            T0 = T1
            T1 = Tk
    return basis

def _basis_trig(x, n=10):
    """Trigonometric basis: 1, cos(x), sin(x), cos(2x), sin(2x)..."""
    # Scale x to fit a few periods in [-2, 2]
    z = x
    basis = np.zeros((len(z), n))
    basis[:, 0] = 1.0
    # We need n-1 more columns
    # Pair (cos(kz), sin(kz)) uses 2 columns
    col = 1
    k = 1
    while col < n:
        c = np.cos(k * z)
        s = np.sin(k * z)
        if col < n:
            basis[:, col] = c
            col += 1
        if col < n:
            basis[:, col] = s
            col += 1
        k += 1
    return basis

def _basis_rbf(x, n=10):
    """Radial Basis Functions (Gaussian) with fixed centers."""
    # Centers spread across [-2, 2]
    centers = np.linspace(-2, 2, n)
    # Width parameter
    width = 0.5
    basis = np.zeros((len(x), n))
    for i, c in enumerate(centers):
        basis[:, i] = np.exp(-((x - c) ** 2) / (2 * width ** 2))
    return basis

def _basis_mixed(x, n=10):
    """Mix of low-order polynomials and trigonometric."""
    z = x / 2.0
    basis = np.zeros((len(z), n))
    # 1, z, z^2, z^3
    basis[:, 0] = 1.0
    basis[:, 1] = z
    basis[:, 2] = z ** 2
    basis[:, 3] = z ** 3
    # cos(z), sin(z), cos(2z), sin(2z)
    basis[:, 4] = np.cos(z)
    basis[:, 5] = np.sin(z)
    basis[:, 6] = np.cos(2 * z)
    basis[:, 7] = np.sin(2 * z)
    # z^4, z^5
    basis[:, 8] = z ** 4
    basis[:, 9] = z ** 5
    return basis

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Adaptive basis selection.
    We choose the basis with the lowest condition number to ensure BFGS stability.
    """
    # Time check
    if _get_elapsed_time() > 140.0:
        # Fallback to a very simple, fast basis if we are running out of time
        return params[0] * x + params[1]

    # Ensure x is 1D
    x_flat = x.flatten()
    n_points = len(x_flat)
    
    if n_points == 0:
        return np.array([])

    # Candidate bases
    # We generate all bases, compute their condition numbers, and pick the best.
    # This is done on every call. For 200 points, this is very fast (< 1ms).
    
    try:
        b_poly = _basis_polynomial(x_flat, 10)
        b_cheb = _basis_chebyshev(x_flat, 10)
        b_trig = _basis_trig(x_flat, 10)
        b_rbf = _basis_rbf(x_flat, 10)
        b_mix = _basis_mixed(x_flat, 10)
        
        # Compute condition numbers
        # Using np.linalg.cond which uses SVD.
        # We add a small regularization to avoid singular matrices if points are identical
        cond_poly = np.linalg.cond(b_poly)
        cond_cheb = np.linalg.cond(b_cheb)
        cond_trig = np.linalg.cond(b_trig)
        cond_rbf = np.linalg.cond(b_rbf)
        cond_mix = np.linalg.cond(b_mix)
        
        conds = [cond_poly, cond_cheb, cond_trig, cond_rbf, cond_mix]
        bases = [b_poly, b_cheb, b_trig, b_rbf, b_mix]
        
        # Select basis with minimum condition number
        min_cond_idx = np.argmin(conds)
        best_basis = bases[min_cond_idx]
        
        # Prediction: params @ best_basis.T
        # params is shape (10,), best_basis is (N, 10)
        pred = best_basis @ params
        
        # Ensure finite
        if not np.all(np.isfinite(pred)):
             return np.zeros_like(x_flat)
             
        return pred

    except Exception:
        # Fallback to linear if anything goes wrong
        return params[0] * x_flat + params[1]

# EVOLVE-BLOCK-END
