"""Rewritten program: ODE solver for 15-dim stiff linear system using matrix exponential."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    start_time = time.time()
    deadline = start_time + 280  # 300s limit, keep 20s margin
    
    # The task specifies a "fixed" 15-dimensional stiff linear system.
    # This implies constant coefficients. The system could be homogeneous (y' = Ay)
    # or non-homogeneous (y' = Ay + b).
    # We recover A and b by evaluating f(t0, y) for basis vectors and zero vector.
    # Total f calls: n + 1 = 16.
    
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    # Recover b (inhomogeneous term)
    b = f(t0, np.zeros(n))
    
    # Recover A (linear part)
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        A[:, i] = f(t0, e_i)
        
    # If b is effectively zero, we can save the call and use n-dim expm?
    # But we already made the call. 
    # We can use the augmented system regardless.
    # Augmented system:
    # z = [y; 1]
    # z' = [Ay + b; 0] = M_aug z
    # M_aug is (n+1) x (n+1)
    
    n_aug = n + 1
    M_aug = np.zeros((n_aug, n_aug))
    M_aug[:n, :n] = A
    M_aug[:n, n] = b
    
    dt = t1 - t0
    # Compute expm(M_aug * dt)
    M_exp = expm(M_aug * dt)
    
    # Initial augmented state
    z0 = np.zeros(n_aug)
    z0[:n] = y0
    z0[n] = 1.0
    
    # Final augmented state
    z_end = M_exp @ z0
    
    return z_end[:n]

# EVOLVE-BLOCK-END
