"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def _extract_matrix(f, t0, y0):
    """
    Extract the system matrix A for a linear ODE y' = A y using exactly n calls.
    
    Strategy: Use y0 as one of the n linearly independent test vectors, and n-1 unit vectors
    for the remaining columns. Specifically:
      - v_0 = y0
      - v_j = e_{j-1} for j = 1, ..., n-1 (i.e., e_0, e_1, ..., e_{n-2})
    
    If these vectors are linearly independent, we can solve A = F @ Y^{-1} where
    Y has columns v_0, ..., v_{n-1} and F has columns f(t, v_i).
    
    If the matrix Y is singular (y0 in span of {e_0, ..., e_{n-2}}), fall back to
    using all n unit vectors.
    """
    n = len(y0)
    y0 = np.asarray(y0, dtype=float)
    
    # Build the test vector matrix Y with columns v_0, ..., v_{n-1}
    Y = np.eye(n)
    Y[:, 0] = y0
    
    # Check if Y is well-conditioned enough
    try:
        cond = np.linalg.cond(Y)
        if cond > 1e12:
            # Fall back to unit vectors
            Y = np.eye(n)
    except np.linalg.LinAlgError:
        Y = np.eye(n)
    
    # Make n calls to extract F
    F = np.empty((n, n))
    for j in range(n):
        F[:, j] = f(t0, Y[:, j])
    
    # Solve A = F @ Y^{-1}
    A = F @ np.linalg.inv(Y)
    return A


def _solve_linear_ode_explicit(A, t0, t1, y0):
    """
    Solve y' = A y using matrix exponential via scipy.
    """
    from scipy.linalg import expm
    dt = t1 - t0
    M = expm(A * dt)
    return M @ y0


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Extract A using n calls
    A = _extract_matrix(f, t0, y0)
    
    # Solve using matrix exponential
    y_end = _solve_linear_ode_explicit(A, t0, t1, y0)
    
    return y_end
# EVOLVE-BLOCK-END
