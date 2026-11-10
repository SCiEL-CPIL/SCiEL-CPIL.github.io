# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, gammaln
from scipy.optimize import minimize

_DEADLINE = time.time() + 610

def _build_quadrature():
    """70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    cos_theta = x_gl
    phi = np.linspace(0, 2 * np.pi, 130, endpoint=False)
    dphi = 2.0 * np.pi / 130
    weights_1d = w_gl * dphi
    return cos_theta, phi, weights_1d

def _compute_harmonics_matrix(cos_theta, phi):
    """
    Compute the matrix X where X[i, j] = Y_j(ct_i, phi_j).
    j ranges from 0 to 20, corresponding to m = -10 to 10.
    """
    N_pts_ct = len(cos_theta)
    N_pts_phi = len(phi)
    N_pts = N_pts_ct * N_pts_phi
    X = np.zeros((N_pts, 21))
    
    ct, ph = np.meshgrid(cos_theta, phi, indexing='ij')
    ct_flat = ct.flatten()
    ph_flat = ph.flatten()
    
    for k, m in enumerate(range(-10, 11)):
        abs_m = abs(m)
        log_N = 0.5 * (np.log(21.0 / (4.0 * np.pi)) + 
                       gammaln(10 - abs_m + 1) - gammaln(10 + abs_m + 1))
        N_m = np.exp(log_N)
        
        P = lpmv(abs_m, 10, ct_flat)
        
        if m == 0:
            Y = N_m * P
        elif m > 0:
            Y = np.sqrt(2.0) * N_m * P * np.cos(m * ph_flat)
        else:
            Y = np.sqrt(2.0) * N_m * P * np.sin(abs_m * ph_flat)
            
        X[:, k] = Y
        
    return X

def _build_spectral_tensors(X, w_flat):
    """
    Build precomputed tensors for spectral-domain L4 norm computation.
    T4[a,b,c,d] = sum_i w_i X[i,a] X[i,b] X[i,c] X[i,d]
    T2[a,b] = sum_i w_i X[i,a] X[i,b]
    """
    N = 21
    # Build T2 first (21x21)
    T2 = X.T @ (w_flat[:, None] * X)
    
    # Build T4 (21x21x21x21) - this is the key tensor
    # T4[a,b,c,d] = sum_i w_i * X[i,a]*X[i,b]*X[i,c]*X[i,d]
    # We can build this efficiently using outer products
    # T4[a,b,c,d] = sum_i w_i * (X[i,a]*X[i,b]) * (X[i,c]*X[i,d])
    # Let M[i,a,b] = w_i * X[i,a] * X[i,b]
    # Then T4[a,b,c,d] = sum_i M[i,a,b] * X[i,c] * X[i,d]
    
    # More efficient: build pairwise products
    # P2[i, a, b] = X[i,a] * X[i,b]  (N_pts x 21 x 21)
    # Then T4[a,b,c,d] = sum_i w_i * P2[i,a,b] * P2[i,c,d]
    
    N_pts = X.shape[0]
    
    # Build P2: shape (N_pts, 21, 21)
    P2 = X[:, :, None] * X[:, None, :]  # (N_pts, 21, 21)
    
    # Weighted: WP2[i, a, b] = w_i * P2[i, a, b]
    WP2 = P2 * w_flat[:, None, None]
    
    # T4[a,b,c,d] = sum_i WP2[i,a,b] * P2[i,c,d]
    # = (WP2 reshaped) @ (P2 reshaped)
    # WP2 shape: (N_pts, 21, 21) -> (N_pts, 441)
    # P2 shape: (N_pts, 21, 21) -> (N_pts, 441)
    WP2_flat = WP2.reshape(N_pts, 441)
    P2_flat = P2.reshape(N_pts, 441)
    
    T4_flat = WP2_flat.T @ P2_flat  # (441, 441)
    T4 = T4_flat.reshape(21, 21, 21, 21)
    
    return T2, T4

def _ratio_and_grad_spectral(c, T2, T4):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient using spectral tensors.
    
    ||f||_2^2 = sum_{a,b} T2[a,b] c_a c_b = c^T T2 c
    ||f||_4^4 = sum_{a,b,c,d} T4[a,b,c,d] c_a c_b c_c c_d
    """
    c = np.asarray(c, dtype=np.float64)
    
    # S2 = c^T T2 c
    S2 = c @ T2 @ c
    
    if S2 < 1e-15:
        return 1e10, np.zeros(21)
    
    # S4 = sum_{a,b,c,d} T4[a,b,c,d] c_a c_b c_c c_d
    # Efficient computation:
    # Let v = T4 @ c @ c (i.e., v[a,b] = sum_{c,d} T4[a,b,c,d] c_c c_d)
    # Then S4 = sum_{a,b} v[a,b] c_a c_b = c^T v c
    # v[a,b] = sum_{c,d} T4[a,b,c,d] c_c c_d
    # This is like contracting T4 with c on the last two indices
    
    # T4 shape: (21,21,21,21)
    # We want v[a,b] = sum_{c,d} T4[a,b,c,d] * c[c] * c[d]
    # = einsum('abcd,c,d->ab', T4, c, c)
    
    v = np.einsum('abcd,c,d->ab', T4, c, c)
    S4 = c @ v @ c
    
    if S4 < 1e-15:
        return 1e10, np.zeros(21)
    
    N2 = np.sqrt(S2)
    N4 = S4 ** 0.25
    R = N4 / N2
    
    # Gradient of R:
    # dR/dc_i = R * (d/dc_i log R)
    # log R = 0.25 log S4 - 0.5 log S2
    # d/dc_i log S4 = (1/S4) * dS4/dc_i
    # d/dc_i log S2 = (1/S2) * dS2/dc_i
    # dS2/dc_i = 2 * (T2 @ c)_i
    # dS4/dc_i = 4 * sum_{a,b,c} T4[i,a,b,c] c_a c_b c_c
    #           = 4 * (T4 contracted with c on 3 indices)
    
    # dS4/dc_i = 4 * einsum('iabc,a,b,c->i', T4, c, c, c)
    dS4_dc = 4.0 * np.einsum('iabc,a,b,c->i', T4, c, c, c)
    dS2_dc = 2.0 * (T2 @ c)
    
    grad_log_R = dS4_dc / S4 - dS2_dc / S2
    grad_R = R * grad_log_R
    
    return R, grad_R

def rudin_problem10() -> np.ndarray:
    """
    Minimize ||f||_4 / ||f||_2 for degree-10 real spherical harmonics.
    Returns 21 coefficients c_m for m = -10..10.
    """
    global _DEADLINE
    
    cos_theta, phi, w_1d = _build_quadrature()
    X = _compute_harmonics_matrix(cos_theta, phi)
    w_flat = np.tile(w_1d, 130)
    
    # Build spectral tensors (one-time cost)
    T2, T4 = _build_spectral_tensors(X, w_flat)
    
    # Symmetrize T2 for numerical stability
    T2 = 0.5 * (T2 + T2.T)
    
    best_coeffs = None
    best_val = np.inf
    
    rng = np.random.default_rng(42)
    
    starts = []
    # Individual m values
    for m in range(-10, 11):
        c = np.zeros(21)
        c[m + 10] = 1.0
        starts.append(c)
    
    # Pairs of m values
    for m1 in range(-10, 11):
        for m2 in range(m1 + 1, 11):
            c = np.zeros(21)
            c[m1 + 10] = 1.0 / np.sqrt(2)
            c[m2 + 10] = 1.0 / np.sqrt(2)
            starts.append(c)
    
    # Random starts
    for _ in range(300):
        c = rng.standard_normal(21)
        c = c / np.linalg.norm(c)
        starts.append(c)
        
    constraint = {'type': 'eq', 'fun': lambda c: np.dot(c, c) - 1.0, 'jac': lambda c: 2.0 * c}
    
    # Use spectral-domain gradient for fast optimization
    for start in starts:
        if time.time() > _DEADLINE - 20:
            break
            
        try:
            res = minimize(
                lambda c: _ratio_and_grad_spectral(c, T2, T4)[0],
                start,
                jac=lambda c: _ratio_and_grad_spectral(c, T2, T4)[1],
                method='SLSQP',
                constraints=[constraint],
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            
            val = _ratio_and_grad_spectral(res.x, T2, T4)[0]
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = res.x.copy()
        except Exception:
            pass

    # Refine best result with more iterations if time permits
    if best_coeffs is not None and time.time() < _DEADLINE - 40:
        try:
            res = minimize(
                lambda c: _ratio_and_grad_spectral(c, T2, T4)[0],
                best_coeffs,
                jac=lambda c: _ratio_and_grad_spectral(c, T2, T4)[1],
                method='SLSQP',
                constraints=[constraint],
                options={'maxiter': 5000, 'ftol': 1e-16}
            )
            val = _ratio_and_grad_spectral(res.x, T2, T4)[0]
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Additional polish pass
    if best_coeffs is not None and time.time() < _DEADLINE - 15:
        try:
            res = minimize(
                lambda c: _ratio_and_grad_spectral(c, T2, T4)[0],
                best_coeffs,
                jac=lambda c: _ratio_and_grad_spectral(c, T2, T4)[1],
                method='SLSQP',
                constraints=[constraint],
                options={'maxiter': 10000, 'ftol': 1e-16}
            )
            val = _ratio_and_grad_spectral(res.x, T2, T4)[0]
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_coeffs = res.x.copy()
        except Exception:
            pass

    if best_coeffs is None:
        best_coeffs = np.zeros(21)
        best_coeffs[10] = 1.0

    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs * (0.01 / max_abs)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs * (1e4 / max_abs)
        
    return best_coeffs

# EVOLVE-BLOCK-END
