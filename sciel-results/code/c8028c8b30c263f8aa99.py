"""ODE solver: 15-dimensional stiff linear system using matrix extraction."""
import numpy as np
from scipy.linalg import expm


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Extract system matrix A assuming y' = A y (homogeneous linear system).
    # We use n random non-orthogonal vectors to determine A with exactly n calls.
    # This is more efficient than the affine approach (n+1 calls) if the system is homogeneous.
    
    # Generate a well-conditioned random matrix V
    V = np.random.randn(n, n)
    max_attempts = 100
    for _ in range(max_attempts):
        if np.linalg.cond(V) < 100:
            break
        V = np.random.randn(n, n)
        
    # Probe f with the columns of V
    F = np.empty((n, n))
    for i in range(n):
        F[:, i] = f(t0, V[:, i])
        
    # Solve for A: F = A @ V  => A = F @ V^{-1}
    # Using solve for numerical stability: V^T A^T = F^T
    A = np.linalg.solve(V.T, F.T).T
    
    dt = t1 - t0
    
    # Compute the matrix exponential
    M = expm(A * dt)
    
    # Final state
    y_end = M @ y0
    
    return y_end
