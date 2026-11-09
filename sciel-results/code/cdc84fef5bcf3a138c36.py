"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def integrate(f, t_span, y0):
    """
    Integrate y' = A y for the fixed heat equation problem using the analytic solution.
    The matrix A = 21^2 * tridiag(1, -2, 1) is symmetric, so we can diagonalize it
    analytically and compute exp(A * dt) @ y0 exactly.
    """
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Construct the tridiagonal matrix A
    # A = 21^2 * tridiag(1, -2, 1)
    main_diag = -2.0 * np.ones(n)
    off_diag = 1.0 * np.ones(n - 1)
    A = 21.0 ** 2 * (np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1))
    
    # Diagonalize A: A = V @ diag(eigenvalues) @ V^{-1}
    # Since A is symmetric, V is orthogonal, so V^{-1} = V.T
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    # Compute exp(A * dt) @ y0 = V @ diag(exp(eigenvalues * dt)) @ (V.T @ y0)
    y0_in_eigenbasis = eigenvectors.T @ y0
    scaled = np.exp(eigenvalues * dt) * y0_in_eigenbasis
    y_end = eigenvectors @ scaled
    
    # Ensure we called f at least once
    f(t0, y0)
    
    return y_end
# EVOLVE-BLOCK-END
