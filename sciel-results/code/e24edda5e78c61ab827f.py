"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import factorial, sqrt, pi

def _hermite_val(n, x):
    """Compute physicists' Hermite polynomial H_n(x) at scalar x."""
    if n == 0:
        return 1.0
    elif n == 1:
        return 2.0 * x
    H_prev = 1.0
    H_curr = 2.0 * x
    for k in range(2, n + 1):
        H_next = 2.0 * x * H_curr - 2.0 * (k - 1) * H_prev
        H_prev = H_curr
        H_curr = H_next
    return H_curr

# Basis indices: 0, 4, 8, 12
indices = [0, 4, 8, 12]

# Orthogonality: int H_n H_m exp(-x^2) dx = sqrt(pi) * 2^n * n! * delta_nm
M_diag = np.array([sqrt(pi) * (2**n) * factorial(n) for n in indices])

# N_ii = int x^2 H_n^2 exp(-x^2) dx = n * M_ii
N_diag = np.array([n * m for n, m in zip(indices, M_diag)])

# Values at x=0 for the constraint g(0)=0
H0_0 = _hermite_val(0, 0)    # 1
H4_0 = _hermite_val(4, 0)    # 12
H8_0 = _hermite_val(8, 0)    # 168
H12_0 = _hermite_val(12, 0)  # 20901120


def _compute_bound_analytic(c0, c4, c8):
    """Compute the uncertainty bound for given coefficients."""
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    coeffs = np.array([c0, c4, c8, c12])
    
    norm_sq = np.sum(coeffs**2 * M_diag)
    if norm_sq <= 1e-12:
        return 1e10
        
    x2_expectation = np.sum(coeffs**2 * N_diag)
    return x2_expectation / norm_sq


def construct():
    """Find optimal coefficients for H0, H4, H8."""
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin before 180s limit
    
    best_val = 1e10
    best_u = None
    
    # Analytic solution via generalized eigenvalue problem
    # The objective is a Rayleigh quotient R(c) = c^T N c / c^T M c
    # where c = [c0, c4, c8, c12] with c12 = a0*c0 + a4*c4 + a8*c8
    # So c = P * u where u = [c0, c4, c8] and P is a 4x3 matrix
    # Then R(u) = u^T (P^T N P) u / u^T (P^T M P) u
    # The minimum is the smallest generalized eigenvalue
    
    try:
        a_vec = np.array([-H0_0/H12_0, -H4_0/H12_0, -H8_0/H12_0])
        P = np.zeros((4, 3))
        P[0, 0] = 1.0
        P[1, 1] = 1.0
        P[2, 2] = 1.0
        P[3, :] = a_vec
        
        M_full = np.diag(M_diag)
        N_full = np.diag(N_diag)
        
        M_proj = P.T @ M_full @ P
        N_proj = P.T @ N_full @ P
        
        from scipy.linalg import eigh
        eigvals, eigvecs = eigh(N_proj, M_proj)
        idx = np.argmin(eigvals)
        best_val = eigvals[idx]
        best_u = eigvecs[:, idx]
        
        # Normalize for consistent scale
        max_comp = np.max(np.abs(best_u))
        if max_comp > 1e-15:
            best_u = best_u / max_comp
            # Ensure c0 is positive for consistency
            if best_u[0] < 0:
                best_u = -best_u
        
        actual_val = _compute_bound_analytic(*best_u)
        if actual_val < best_val:
            best_val = actual_val
    except Exception:
        pass
    
    # Fallback if eigenvalue approach failed
    if best_u is None:
        best_u = np.array([1.0, 0.0, 0.0])
        best_val = _compute_bound_analytic(*best_u)
    
    # The generalized eigenvalue problem gives the exact global minimum
    # of the Rayleigh quotient over the 3-dimensional subspace defined by
    # the constraint g(0) = 0. No further optimization is needed.
    
    # Verify and return
    final_val = _compute_bound_analytic(*best_u)
    
    return best_u
# EVOLVE-BLOCK-END
