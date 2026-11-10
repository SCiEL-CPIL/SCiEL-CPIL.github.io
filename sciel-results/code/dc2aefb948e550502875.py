# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv
import time


def _build_quadrature():
    """Build the 70x130 quadrature grid and compute Y_m values."""
    k = 10
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights in [-1, 1]
    x, w = np.polynomial.legendre.leggauss(n_theta)
    
    # Equally spaced phi
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    
    # Create meshgrid
    X, Phi = np.meshgrid(x, phi, indexing='ij')  # shape (n_theta, n_phi)
    W = np.outer(w, np.ones(n_phi)) / n_phi     # shape (n_theta, n_phi)
    
    # Compute Y_m for m = -10..10
    # Y_0 = N_0 * P_10^0(cos theta)
    # Y_m = sqrt(2) * N_m * P_10^m(cos theta) * cos(m*phi) for m > 0
    # Y_m = sqrt(2) * N_|m| * P_10^|m|(cos theta) * sin(|m|*phi) for m < 0
    # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    
    Y = np.zeros((21, n_theta, n_phi), dtype=np.float64)
    
    for idx, m in enumerate(range(-k, k + 1)):
        abs_m = abs(m)
        # N_m
        # (10-abs_m)! / (10+abs_m)!
        log_ratio = 0.0
        for j in range(10 - abs_m + 1, 10 + abs_m + 1):
            log_ratio -= np.log(j)
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * np.exp(log_ratio))
        
        # Associated Legendre with Condon-Shortley phase
        # lpmv(m, l, x) includes the Condon-Shortley phase (-1)^m
        P = lpmv(abs_m, k, X)  # shape (n_theta, n_phi) but P depends only on X
        
        if m == 0:
            Y[idx] = N_m * P
        elif m > 0:
            Y[idx] = np.sqrt(2.0) * N_m * P * np.cos(m * Phi)
        else:
            Y[idx] = np.sqrt(2.0) * N_m * P * np.sin(abs_m * Phi)
    
    return Y, W, x, phi


def _compute_R(coeffs, Y, W):
    """Compute R = ||f||_4 / ||f||_2 using precomputed quadrature."""
    # f = sum_m c_m Y_m
    f = np.tensordot(coeffs, Y, axes=(0, 0))  # shape (n_theta, n_phi)
    
    # ||f||_2^2 = sum W * f^2
    f2 = f * f
    norm2_sq = np.sum(W * f2)
    
    if norm2_sq <= 0:
        return np.inf
    
    # ||f||_4^4 = sum W * f^4
    f4 = f2 * f2
    norm4_4 = np.sum(W * f4)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))  # (f^4)^(1/4) = |f|, so norm4 = (sum W * |f|^4)^(1/4)
    
    return norm4 / norm2


def rudin_problem10() -> np.ndarray:
    k = 10
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, keep margin
    
    # Build quadrature
    Y, W, x, phi = _build_quadrature()
    n_theta, n_phi = Y.shape[1:]
    
    # Precompute for speed: flatten Y to (21, N) where N = n_theta * n_phi
    Y_flat = Y.reshape(21, -1)  # (21, N)
    W_flat = W.reshape(-1)       # (N,)
    
    def objective(c):
        """Minimize ||f||_4 subject to ||c||_2 = 1 (handled by constraint)."""
        f = Y_flat @ c  # (N,)
        f2 = f * f
        norm2_sq = np.dot(W_flat, f2)
        if norm2_sq <= 0:
            return 1e10
        norm4_4 = np.dot(W_flat, f2 * f2)
        norm4 = np.sqrt(np.sqrt(norm4_4))
        norm2 = np.sqrt(norm2_sq)
        return norm4 / norm2
    
    def objective_scaled(c):
        """Alternative: minimize ||f||_4^4 / ||f||_2^4 to avoid sqrt."""
        f = Y_flat @ c
        f2 = f * f
        norm2_sq = np.dot(W_flat, f2)
        if norm2_sq <= 0:
            return 1e10
        norm4_4 = np.dot(W_flat, f2 * f2)
        return norm4_4 / (norm2_sq * norm2_sq)
    
    # We'll minimize objective_scaled which is R^4, scale-invariant.
    # Use constraint ||c||^2 = 1
    
    def constraint_eq(c):
        return np.dot(c, c) - 1.0
    
    constraints = [{'type': 'eq', 'fun': constraint_eq}]
    
    # Generate diverse initial guesses
    rng = np.random.default_rng(42)
    
    starts = []
    
    # 1. All ones, normalized
    c0 = np.ones(21)
    c0 /= np.linalg.norm(c0)
    starts.append(c0)
    
    # 2. Random starts
    for _ in range(30):
        c = rng.standard_normal(21)
        c /= np.linalg.norm(c)
        starts.append(c)
    
    # 3. Single-m starts
    for m_idx in range(21):
        c = np.zeros(21)
        c[m_idx] = 1.0
        starts.append(c)
    
    # 4. Cos/sin pairs (m=1..10, use +m and -m)
    for m in range(1, 11):
        c = np.zeros(21)
        c[m + 10] = 1.0 / np.sqrt(2)
        c[-m + 10] = 1.0 / np.sqrt(2)
        starts.append(c)
        
        c2 = np.zeros(21)
        c2[m + 10] = 1.0 / np.sqrt(2)
        c2[-m + 10] = -1.0 / np.sqrt(2)
        starts.append(c2)
    
    # 5. Alternating signs
    c = np.ones(21)
    c[::2] = -1.0
    c /= np.linalg.norm(c)
    starts.append(c)
    
    # 6. Sparse starts
    for _ in range(10):
        idx = rng.choice(21, size=5, replace=False)
        c = np.zeros(21)
        c[idx] = rng.standard_normal(5)
        c /= np.linalg.norm(c)
        starts.append(c)
    
    best_c = None
    best_val = np.inf
    
    for i, c0 in enumerate(starts):
        if time.time() > deadline - 30.0:
            break
        
        try:
            res = minimize(
                objective_scaled,
                c0,
                method='L-BFGS-B',
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            c_res = res.x
            # Re-normalize
            c_res /= np.linalg.norm(c_res)
            
            val = objective_scaled(c_res)
            if val < best_val:
                best_val = val
                best_c = c_res.copy()
        except Exception:
            pass
    
    # Second phase: perturb the best solution and re-optimize
    if best_c is not None and time.time() < deadline - 60.0:
        for _ in range(20):
            if time.time() > deadline - 30.0:
                break
            c0 = best_c + 0.1 * rng.standard_normal(21)
            c0 /= np.linalg.norm(c0)
            
            try:
                res = minimize(
                    objective_scaled,
                    c0,
                    method='L-BFGS-B',
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                c_res = res.x
                c_res /= np.linalg.norm(c_res)
                
                val = objective_scaled(c_res)
                if val < best_val:
                    best_val = val
                    best_c = c_res.copy()
            except Exception:
                pass
    
    # Third phase: even smaller perturbations
    if best_c is not None and time.time() < deadline - 30.0:
        for _ in range(10):
            if time.time() > deadline - 10.0:
                break
            c0 = best_c + 0.01 * rng.standard_normal(21)
            c0 /= np.linalg.norm(c0)
            
            try:
                res = minimize(
                    objective_scaled,
                    c0,
                    method='L-BFGS-B',
                    constraints=constraints,
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                
                c_res = res.x
                c_res /= np.linalg.norm(c_res)
                
                val = objective_scaled(c_res)
                if val < best_val:
                    best_val = val
                    best_c = c_res.copy()
            except Exception:
                pass
    
    if best_c is None:
        return np.ones(21, dtype=np.float64)
    
    # Validate
    best_c = np.asarray(best_c, dtype=np.float64)
    if len(best_c) != 21:
        return np.ones(21, dtype=np.float64)
    if np.any(~np.isfinite(best_c)):
        return np.ones(21, dtype=np.float64)
    if np.max(np.abs(best_c)) < 1e-3 or np.max(np.abs(best_c)) > 1e5:
        return np.ones(21, dtype=np.float64)
    
    return best_c


# EVOLVE-BLOCK-END
