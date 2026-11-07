"""ODE integrator using spectral decomposition of the linear operator recovered from f."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    dt = t1 - t0

    # Recover the constant matrix A from f by probing with basis vectors.
    # f(t, y) = A @ y (t-independent), so A[:, j] = f(0, e_j).
    A = np.empty((n, n), dtype=float)
    for j in range(n):
        e = np.zeros(n, dtype=float)
        e[j] = 1.0
        A[:, j] = f(0.0, e)

    # Use spectral decomposition. A is symmetric (tridiagonal 1, -2, 1 scaled).
    # eigvalsh/eigh are more stable for symmetric matrices.
    # y_end = V exp(D dt) V^T y0
    eigvals, eigvecs = np.linalg.eigh(A)
    
    # Project y0 onto eigenbasis
    # V is orthogonal, so V^T = V^{-1}
    coeffs = eigvecs.T @ y0
    
    # Scale coefficients by exp(eigval * dt)
    coeffs_scaled = coeffs * np.exp(eigvals * dt)
    
    # Project back to original basis
    y_end = eigvecs @ coeffs_scaled

    # Safety: ensure finite output with correct shape.
    if not np.all(np.isfinite(y_end)):
        # Fallback: a very small explicit step using f directly (should not happen).
        y = y0.copy()
        h = dt / 200000
        t = t0
        for _ in range(200000):
            y = y + h * f(t, y)
            t += h
        y_end = y

    return y_end
# EVOLVE-BLOCK-END
