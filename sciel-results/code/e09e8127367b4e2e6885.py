"""Rewritten program: degree-5 polynomial closure for unresolved tendency."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _polyval5(x, c):
    """Evaluate degree-5 polynomial with coefficients c[0..5] in ascending powers."""
    # Horner's method for stability
    r = c[5]
    for i in range(4, -1, -1):
        r = r * x + c[i]
    return r


def _build_vandermonde5(x):
    """Build 5xN Vandermonde matrix (ascending powers) for stable least squares."""
    x = np.asarray(x, dtype=np.float64)
    # Center and scale for numerical stability
    xc = x - x.mean()
    # Build columns: 1, xc, xc^2, xc^3, xc^4, xc^5
    V = np.empty((6, x.size), dtype=np.float64)
    V[0] = 1.0
    V[1] = xc
    V[2] = xc * xc
    V[3] = V[2] * xc
    V[4] = V[3] * xc
    V[5] = V[4] * xc
    return V, xc


def construct(training_x, training_subgrid_tendency):
    """
    Fit a shared degree-5 polynomial to the unresolved fast-scale forcing.
    
    Parameters:
        training_x: flattened array of slow-state values
        training_subgrid_tendency: flattened array of corresponding subgrid tendencies
    
    Returns:
        numpy array of 6 finite real coefficients in ascending powers of x
    """
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()

    # Ensure finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = x.size
    if n == 0:
        return np.zeros(6, dtype=np.float64)

    # Center x for numerical stability
    x_mean = x.mean()
    x_std = x.std()
    if x_std < 1e-12:
        x_std = 1.0

    xs = (x - x_mean) / x_std

    # Build Vandermonde in scaled variable
    V = np.empty((6, n), dtype=np.float64)
    V[0] = 1.0
    V[1] = xs
    V[2] = xs * xs
    V[3] = V[2] * xs
    V[4] = V[3] * xs
    V[5] = V[4] * xs

    # Solve least squares: V @ c = y
    # Use SVD for robustness
    try:
        # Compute via lstsq
        c_scaled, residuals, rank, sv = np.linalg.lstsq(V, y, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        try:
            VtV = V.T @ V
            Vty = V.T @ y
            c_scaled = np.linalg.solve(VtV + 1e-12 * np.eye(6), Vty)
        except np.linalg.LinAlgError:
            c_scaled = np.zeros(6, dtype=np.float64)

    # Convert from scaled variable back to original x
    # If xs = (x - x_mean) / x_std, then x = x_std * xs + x_mean
    # We need to express sum_{i=0}^{5} c_scaled[i] * xs^i in terms of x
    # xs = (x - x_mean) / x_std
    # So we need to expand and collect coefficients in powers of x

    # Use binomial expansion: (x - x_mean)^i / x_std^i
    # coefficient of x^j in (x - x_mean)^i is C(i,j) * (-x_mean)^(i-j)
    
    c_original = np.zeros(6, dtype=np.float64)
    
    for i in range(6):
        ci = c_scaled[i]
        if abs(ci) < 1e-15:
            continue
        # Expand (x - x_mean)^i / x_std^i
        # = sum_{j=0}^{i} C(i,j) * x^j * (-x_mean)^(i-j) / x_std^i
        for j in range(i + 1):
            # C(i, j)
            from math import comb
            binom = comb(i, j)
            term = ci * binom * ((-x_mean) ** (i - j)) / (x_std ** i)
            c_original[j] += term

    # Check for non-finite values
    if not np.all(np.isfinite(c_original)):
        c_original = np.nan_to_num(c_original, nan=0.0, posinf=0.0, neginf=0.0)

    # Verify within deadline
    elapsed = time.time() - t0
    if elapsed > deadline - t0:
        # Shouldn't happen, but return what we have
        pass

    return c_original
# EVOLVE-BLOCK-END
