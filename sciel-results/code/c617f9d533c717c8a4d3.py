"""ODE solver for 10x10 2D heat equation using Crank-Nicolson with precomputed Jacobian."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg as la
import time


def _build_heat_matrix(n=10):
    """Build the 2D Laplacian matrix for a 10x10 grid with Dirichlet BCs.
    The state vector y is flattened row-major (y[i*10+j] corresponds to grid point (i,j)).
    The Laplacian for interior points (with zero Dirichlet BCs) has:
      -4 on the diagonal
      +1 for 4-neighbors (up, down, left, right)
    """
    n2 = n * n
    J = np.zeros((n2, n2))
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            J[idx, idx] = -4.0
            if i > 0:
                J[idx, (i - 1) * n + j] = 1.0
            if i < n - 1:
                J[idx, (i + 1) * n + j] = 1.0
            if j > 0:
                J[idx, i * n + (j - 1)] = 1.0
            if j < n - 1:
                J[idx, i * n + (j + 1)] = 1.0
    return J


def _estimate_jacobian(f, t, y):
    """Estimate Jacobian of f(t, y) via forward differences."""
    n = len(y)
    J = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1e-7 * max(1.0, abs(y[j]))
        J[:, j] = (f(t, y + e) - f(t, y)) / e[j]
    return J


def integrate(f, t_span, y0):
    """Solve y' = f(t, y) using Crank-Nicolson with precomputed Jacobian.
    For the heat equation, f is linear and time-independent, so we can
    precompute the LU factorization of (I - 0.5*h*J) and reuse it.
    """
    t0, t1 = t_span
    y = np.array(y0, dtype=float)
    n = len(y)
    
    # Deadline guard: stop after 300s to leave safety margin
    deadline = time.time() + 290.0
    
    # Estimate Jacobian at initial state
    J = _estimate_jacobian(f, t0, y)
    
    # Choose number of steps - Crank-Nicolson is 2nd order accurate
    # For heat equation on 10x10, we need reasonable accuracy
    # Let's use a moderate number of steps
    nsteps = 500
    h = (t1 - t0) / nsteps
    
    # Precompute LU factorization of A = I - 0.5*h*J
    I = np.eye(n)
    A = I - 0.5 * h * J
    B = I + 0.5 * h * J
    
    lu, piv = la.lu_factor(A)
    
    t = t0
    for step in range(nsteps):
        # Check deadline
        if time.time() > deadline:
            break
        
        # Crank-Nicolson: (I - 0.5*h*J) y_{n+1} = (I + 0.5*h*J) y_n
        rhs = B @ y
        y = la.lu_solve((lu, piv), rhs)
        t += h
    
    return y
# EVOLVE-BLOCK-END
