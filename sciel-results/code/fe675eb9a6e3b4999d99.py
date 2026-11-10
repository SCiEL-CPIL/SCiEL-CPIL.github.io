# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize


def _precompute_tensor():
    """
    Precompute the 4th-order moment tensor T_{ijkl} = int Y_i Y_j Y_k Y_l dOmega.
    Also returns the basis matrix for potential verification.
    """
    k = 10
    N_theta = 70
    N_phi = 130
    N = 2 * k + 1
    
    theta_grid, w_theta = np.polynomial.legendre.leggauss(N_theta)
    phi_grid = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    
    # Build basis matrix M of shape (N_theta * N_phi, N)
    # M[n, i] = Y_i(theta_n, phi_n)
    M = np.zeros((N_theta * N_phi, N), dtype=np.float64)
    
    idx = 0
    for i_th in range(N_theta):
        x = theta_grid[i_th]
        w_th = w_theta[i_th]
        for j_ph in range(N_phi):
            phi = phi_grid[j_ph]
            for m_idx, m in enumerate(range(-k, k + 1)):
                abs_m = abs(m)
                P = lpmv(abs_m, k, x)
                
                if abs_m == 0:
                    fact_ratio = 1.0
                else:
                    prod = 1.0
                    for j in range(10 - abs_m + 1, 10 + abs_m + 1):
                        prod *= j
                    fact_ratio = 1.0 / prod
                
                N_m = np.sqrt(21.0 / (4.0 * np.pi) * fact_ratio)
                
                if m == 0:
                    val = N_m * P
                elif m > 0:
                    val = np.sqrt(2.0) * N_m * P * np.cos(m * phi)
                else:
                    val = np.sqrt(2.0) * N_m * P * np.sin(abs_m * phi)
                
                M[idx, m_idx] = val
            idx += 1
    
    # Weights for quadrature:
    # The integral is sum_{th, ph} w_th * (2*pi/N_phi) * Y...
    # Let w_eff be the weight for each point in M.
    w_eff = np.repeat(w_theta, N_phi) * (2.0 * np.pi / N_phi)
    
    # Compute T_{ijkl} = sum_n w_eff[n] * M[n,i] * M[n,j] * M[n,k] * M[n,l]
    # Using einsum for efficiency
    T = np.einsum('n,ni,nj,nk,nl->ijkl', w_eff, M, M, M, M, optimize=True)
    
    return T


def _compute_R_from_tensor(c, T):
    """
    Compute R = ||f||_4 / ||f||_2 from tensor T.
    ||f||_2^2 = sum c_i^2 (since basis is orthonormal)
    ||f||_4^4 = sum T_{ijkl} c_i c_j c_k c_l
    """
    norm2_sq = np.dot(c, c)
    if norm2_sq == 0:
        return np.inf
    
    # ||f||_4^4
    # Q = c_i c_j c_k c_l T_{ijkl}
    # Efficiently:
    # Q = sum_i c_i ( sum_j c_j ( sum_k c_k ( sum_l T_{ijkl} c_l ) ) )
    # Or using tensordot:
    # tmp1 = np.tensordot(T, c, axes=([3], [0])) -> (21,21,21)
    # tmp2 = np.tensordot(tmp1, c, axes=([2], [0])) -> (21,21)
    # tmp3 = np.tensordot(tmp2, c, axes=([1], [0])) -> (21)
    # Q = np.dot(c, tmp3)
    
    tmp1 = np.tensordot(T, c, axes=([3], [0]))
    tmp2 = np.tensordot(tmp1, c, axes=([2], [0]))
    tmp3 = np.tensordot(tmp2, c, axes=([1], [0]))
    norm4_4 = np.dot(c, tmp3)
    
    if norm4_4 < 0:
        norm4_4 = 0.0
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    return norm4 / norm2


def _compute_grad_R_from_tensor(c, T):
    """
    Compute gradient of R with respect to c.
    R = N4 / N2
    dR/dc_m = (1/N2) dN4/dc_m - (N4/N2^2) dN2/dc_m
    
    N2 = sum c_i^2 -> dN2/dc_m = 2 c_m
    N4 = sum T_{ijkl} c_i c_j c_k c_l
    dN4/dc_m = 4 sum_{ijk} T_{mijk} c_i c_j c_k (by symmetry of T)
    """
    norm2_sq = np.dot(c, c)
    if norm2_sq == 0:
        return np.zeros_like(c)
    
    # dN4/dc
    # grad_N4_m = 4 * sum_{ijk} T_{mijk} c_i c_j c_k
    # Using tensordot:
    # tmp1 = np.tensordot(T, c, axes=([3], [0])) -> (21,21,21) T_{mij k} c_k -> T_{mij}
    # Wait, T_{ijkl}. Contract l with c_l.
    tmp1 = np.tensordot(T, c, axes=([3], [0])) # Shape (21, 21, 21) T_{mij}
    tmp2 = np.tensordot(tmp1, c, axes=([2], [0])) # Shape (21, 21) T_{mi}
    grad_N4 = 4.0 * np.tensordot(tmp2, c, axes=([1], [0])) # Shape (21)
    
    norm2 = np.sqrt(norm2_sq)
    norm4_4 = np.dot(c, grad_N4) / 4.0 # Reuse grad_N4 for N4? No, N4 = 1/4 c . grad_N4
    # Actually N4 = sum c_i dN4/dc_i / 4
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    coeff1 = 1.0 / norm2
    coeff2 = norm4 / (norm2_sq * norm2) # N4 / N2^3 ?
    # R = N4 / N2
    # dR = (N2 dN4 - N4 dN2) / N2^2
    # dR/dc_m = (N2 grad_N4_m - N4 * 2 c_m) / N2^2
    # = grad_N4_m / N2 - 2 N4 c_m / N2^2
    
    grad = grad_N4 / norm2 - (2.0 * norm4_4 / (norm2_sq * norm2)) * c
    
    return grad


def _optimize_with_lbfgs(c0, T, max_iter=5000):
    """
    Refine using L-BFGS-B. Note: L-BFGS-B does not handle equality constraints well,
    but we can project or just run it and re-normalize?
    Better: Use SLSQP or just manual gradient descent with projection.
    However, for 21D, we can use 'L-BFGS-B' on the sphere if we penalize ||c||^2 - 1?
    No, let's use a simple projected gradient descent or Newton.
    Actually, scipy's 'L-BFGS-B' is for bounds.
    Let's use 'Nelder-Mead' or 'Powell' which are derivative-free and robust,
    or implement a simple projected gradient step.
    
    Given the speed of _compute_R_from_tensor, we can afford many iterations.
    Let's use a simple gradient descent with line search and projection.
    """
    c = c0.copy()
    c /= np.linalg.norm(c)
    
    for _ in range(max_iter):
        R = _compute_R_from_tensor(c, T)
        grad = _compute_grad_R_from_tensor(c, T)
        
        # Project gradient onto tangent space
        grad_proj = grad - np.dot(grad, c) * c
        
        # Simple step
        alpha = 0.1
        # Line search
        best_alpha = alpha
        best_R = R
        for a in [0.1, 0.05, 0.01, 0.005, 0.001]:
            c_new = c - a * grad_proj
            c_new_norm = np.linalg.norm(c_new)
            if c_new_norm > 0:
                c_new /= c_new_norm
                R_new = _compute_R_from_tensor(c_new, T)
                if R_new < best_R:
                    best_R = R_new
                    best_alpha = a
                    c = c_new
        
        if best_alpha < 1e-6:
            break
            
    return c, _compute_R_from_tensor(c, T)


def _multi_start_optimization_tensor(T, n_random=50):
    """
    Multi-start optimization using the analytic tensor.
    """
    best_c = None
    best_R = np.inf
    
    # Deterministic starts
    starts = []
    starts.append(np.ones(21) / np.sqrt(21.0))
    starts.append(np.array([(-1)**(i) for i in range(21)], dtype=float) / np.sqrt(21.0))
    
    # Random starts
    np.random.seed(42)
    for _ in range(n_random):
        c = np.random.randn(21)
        c /= np.linalg.norm(c)
        starts.append(c)
        
    # Also try single basis vectors
    for i in range(21):
        c = np.zeros(21)
        c[i] = 1.0
        starts.append(c)
        
    for c_start in starts:
        c_refined, R_refined = _optimize_with_lbfgs(c_start, T)
        if R_refined < best_R:
            best_R = R_refined
            best_c = c_refined.copy()
            
    return best_c, best_R


def rudin_problem10() -> np.ndarray:
    """
    Rudin problem for polynomials on the 2-sphere. Find the real coefficients
    (c_m) for m = -k..k of a real degree-k spherical harmonic

        f = sum_{m=-k}^{k} c_m Y_k^m

    on the round unit sphere S^2 that MINIMIZE the norm ratio

        ||f||_{L^4(S^2)} / ||f||_{L^2(S^2)}

    (norms taken with the standard surface measure dOmega of total mass 4*pi).
    Here the degree is fixed to k = 10, so the returned vector has length
    2*k + 1 = 21.

    Returns:
        coeffs: np.ndarray of shape (21,) with the real coefficients
            c_m, ordered m = -10, -9, ..., 0, ..., 9, 10.
    """
    # Precompute tensor
    T = _precompute_tensor()
    
    # Multi-start optimization
    best_c, best_R = _multi_start_optimization_tensor(T, n_random=50)
    
    if best_c is None:
        best_c = np.ones(21) / np.sqrt(21.0)
    
    # Scale the result to be within valid range
    c = best_c.copy()
    max_abs = np.max(np.abs(c))
    if max_abs < 1e-3:
        c = c * (1.0 / max_abs)
    elif max_abs > 1e5:
        c = c * (1e5 / max_abs)
    
    # Final validation
    if len(c) != 21:
        c = np.ones(21)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        c = np.ones(21)
    max_abs = np.max(np.abs(c))
    if max_abs < 1e-3:
        c = c * (1.0 / max_abs)
    elif max_abs > 1e5:
        c = c * (1e5 / max_abs)
    
    return c


# EVOLVE-BLOCK-END
