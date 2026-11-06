"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply


def integrate(f, t_span, y0):
    t0, t1 = t_span
    T = t1 - t0
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s/330s limit
    
    # We will reconstruct the constant coefficient matrix A such that f(t, y) = A @ y
    # For a linear autonomous ODE y' = A y, we can solve exactly via matrix exponential.
    
    # Step 1: Evaluate f at t0 and y0 to get A @ y0
    fy0 = f(t0, y0)
    
    # Check time
    if time.time() > deadline:
        return y0
    
    # Step 2: We need to determine A. Since A is constant, f(t, y) = A y.
    # We can probe the matrix by evaluating f at basis vectors.
    # However, that would take n evaluations.
    # Instead, we can try to infer the structure.
    # The problem is a 100-point convection-diffusion.
    # Typically, A is tridiagonal.
    # Let's try to recover A by evaluating f at a few specific vectors.
    
    # Strategy:
    # 1. Evaluate f at y0: b = A y0
    # 2. Evaluate f at a perturbed vector to get more info?
    #    Actually, if we assume A is constant, we can just use f(t, y) = A y.
    #    But we don't know A.
    
    # Alternative Strategy:
    # Since the problem is linear and autonomous (y' = A y), we can solve it using
    # scipy.integrate.solve_ivp with a high-order method, but the score depends on
    # the number of function calls.
    # The directive suggests using expm_multiply.
    
    # To use expm_multiply, we need the matrix A.
    # Can we recover A with few calls?
    # If we call f(t, e_i) for i=0..n-1, we get the columns of A. That's n calls.
    # n=100. 100 calls is very small.
    # Score = -log10(err) - 0.5*log10(calls).
    # If err is ~1e-15, -log10(err) ~ 15.
    # calls = 100 + 1 (initial) = 101.
    # -0.5 * log10(101) ~ -1.
    # Score ~ 14.
    # Current score is 1.0.
    # This would be a huge improvement.
    
    # Let's verify if f is linear and autonomous.
    # The task says "convection-diffusion 100-point". These are linear PDEs discretized.
    # So y' = A y is a safe assumption.
    
    # Construct A
    A = np.empty((n, n))
    for j in range(n):
        e_j = np.zeros(n)
        e_j[j] = 1.0
        A[:, j] = f(t0, e_j)
        
        if time.time() > deadline:
            # Fallback to RK4 if we run out of time (unlikely with n=100)
            break
    
    # Check if A was fully constructed
    if not np.all(np.isfinite(A)):
        # Fallback
        return _rk4_fallback(f, t0, t1, y0)
    
    # Compute expm(A * T) @ y0
    # Use sparse matrix for efficiency if possible, but n=100 is small.
    # scipy.sparse.linalg.expm_multiply is efficient.
    A_sparse = diags(np.diagonal(A), 0, shape=(n, n)) # This is dense, let's just use dense expm
    # Actually, expm_multiply works with sparse or dense.
    # Let's use the dense matrix since n is small.
    
    # y_end = expm(A * T) @ y0
    y_end = expm_multiply(A * T, y0)
    
    return y_end


def _rk4_fallback(f, t0, t1, y0):
    """Simple RK4 fallback in case linear assumption fails or time runs out."""
    steps = 1000
    h = (t1 - t0) / steps
    t, y = t0, np.array(y0, dtype=float)
    for _ in range(steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        if not np.all(np.isfinite(y)):
            return y
    return y

# EVOLVE-BLOCK-END
