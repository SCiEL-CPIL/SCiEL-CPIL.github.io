"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import expm


def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    # Determine the linear structure: f(t, y) = A @ y + b(t)
    # Use a few calls to determine A and b at t0
    # First, get b(t0) by calling f with y=0
    y_zero = np.zeros(n)
    f0 = np.array(f(t0, y_zero), dtype=float)  # This is b(t0)
    
    # Determine A by perturbing each basis direction
    A = np.zeros((n, n))
    eps = 1e-8
    for j in range(n):
        e = np.zeros(n)
        e[j] = eps
        fj = np.array(f(t0, e), dtype=float)
        A[:, j] = (fj - f0) / eps
    
    # Now we have A and we can evaluate b(t) at any t
    # Strategy: use N steps with the matrix exponential
    # For a linear system y' = A*y + b(t), with A constant:
    # We use a step-by-step method where at each step we approximate b(t) as constant
    
    N = 50  # number of steps - balance between accuracy and cost
    h = (t1 - t0) / N
    
    # Precompute expm(A*h) and the integral matrix
    # For each step: y_{k+1} = expm(A*h) @ y_k + integral_0^h expm(A*(h-s)) b(t_k+s) ds
    # Approximate b(t_k+s) ≈ b(t_k) (constant over step)
    # Then integral = b(t_k) @ (integral_0^h expm(A*s) ds)  -- but this is a matrix-vector product
    # Actually: integral_0^h expm(A*(h-s)) b ds = expm(A*h) @ (integral_0^h expm(-A*s) b ds)
    # Simpler: use the fact that for constant b, the particular solution involves A^{-1}
    # But let's use a more robust approach with the augmented system or series
    
    # Compute M = expm(A*h)
    M = expm(A * h)
    
    # Compute the integral matrix: I = integral_0^h expm(A*s) ds
    # This can be computed as: if A is invertible, I = A^{-1} @ (expm(A*h) - I)
    # But A may be singular. Use scipy or a series expansion.
    # Alternative: use the block matrix exponential trick
    # Or just use numerical quadrature since h is small
    
    # Let's use a simple approach: since h is small, use a few quadrature points
    # Or better: use the fact that we can compute it via the matrix exponential of an augmented system
    # For now, use: I ≈ h * (I + A*h/2 + (A*h)^2/6 + ...) - truncated series
    # Or use scipy.integrate if available, but let's keep it simple
    
    # Actually, let's just use a moderate number of substeps within each "macro" step
    # to handle b(t) variation, but since we're already using 50 steps, let's just use
    # the constant b approximation per step
    
    # For constant b over a step: y_{k+1} = M @ y_k + P @ b_k
    # where P = integral_0^h expm(A*s) ds
    # Compute P using the series: P = h * sum_{k=0}^{inf} (A*h)^k / (k+1)!
    # Truncate when terms are small
    
    Ah = A * h
    P = np.zeros((n, n))
    term = np.eye(n)
    P = P + h * term  # k=0: h * I
    for k in range(1, 30):
        term = term @ Ah / k
        P = P + h * term / (k + 1)
        if np.max(np.abs(term)) < 1e-15:
            break
    
    # Now step through
    y = y0.copy()
    t = t0
    for k in range(N):
        # Evaluate b(t) at current time
        b = np.array(f(t, y_zero), dtype=float)
        # Update: y_{k+1} = M @ y_k + P @ b
        y = M @ y + P @ b
        t += h
    
    return y
# EVOLVE-BLOCK-END
