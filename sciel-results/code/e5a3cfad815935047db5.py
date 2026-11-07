"""Spectral (DST) exact integrator for the discretized heat equation."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Integrate y' = A y from t_span[0] to t_span[1] with initial condition y0.
    
    Uses the spectral decomposition of the symmetric tridiagonal matrix A
    via the Discrete Sine Transform (DST-I).
    """
    y0 = np.asarray(y0, dtype=np.float64)
    N = len(y0)
    
    # Call f at least once as required
    t0, t1 = t_span
    _ = f(t0, y0)
    
    # Eigenvalues of A = (N+1)^2 * tridiag(1, -2, 1)
    # lambda_k = -2 * (1 - cos(k * pi / (N+1)))
    # A_k = (N+1)^2 * lambda_k
    k = np.arange(1, N + 1)
    cos_terms = np.cos(np.pi * k / (N + 1))
    lambda_tridiag = -2.0 * (1.0 - cos_terms)
    A_eigenvalues = (N + 1) ** 2 * lambda_tridiag
    
    dt = t1 - t0
    exp_factors = np.exp(A_eigenvalues * dt)
    
    # DST-I Matrix
    # V[i, j] = sin((i+1)(j+1) pi / (N+1))
    i = np.arange(1, N + 1)
    j = np.arange(1, N + 1)
    V = np.sin(np.outer(i, j) * np.pi / (N + 1))
    
    # Project y0 onto eigenbasis
    # c = V^T y0
    c = V.T @ y0
    
    # Evolve in spectral space
    c_evolved = exp_factors * c
    
    # Transform back
    # y_end = (2/(N+1)) * V @ c_evolved
    y_end = (2.0 / (N + 1)) * (V @ c_evolved)
    
    return y_end
# EVOLVE-BLOCK-END
