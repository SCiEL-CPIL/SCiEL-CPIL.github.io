"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize
import time

# Precompute Gauss-Laguerre quadrature nodes (higher accuracy)
_N_QUAD = 200
_X_QUAD, _W_QUAD = np.polynomial.laguerre.laggauss(_N_QUAD)

def _compute_c4_and_grad(roots):
    """
    Compute the C4 upper bound and its analytical gradient for given double-root positions.
    Uses Gauss-Laguerre quadrature. C4 = int(P^2 * w) / (int(P * w))^2
    where P(x) = prod_i (x - r_i)^2 and w(x) = e^{-x}.
    """
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf, np.zeros_like(roots)
    
    x = _X_QUAD
    w = _W_QUAD
    
    # diff[i,j] = x_i - r_j, shape (n_quad, 12)
    diff = x[:, np.newaxis] - roots[np.newaxis, :]
    
    # log|diff| for stability; use log(|diff|) carefully
    log_abs_diff = np.log(np.abs(diff) + 1e-300)
    log_P = 2.0 * np.sum(log_abs_diff, axis=1)  # shape (n_quad,)
    
    # P = exp(log_P), P2 = exp(2*log_P)
    P = np.exp(log_P)
    P2 = np.exp(2.0 * log_P)
    
    int_P = np.sum(w * P)
    int_P2 = np.sum(w * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
        return np.inf, np.zeros_like(roots)
    
    c4 = int_P2 / (int_P * int_P)
    
    # Analytical gradient: dC4/dr_k
    # C4 = A / B^2 where A = int_P2, B = int_P
    # dA/dr_k = sum_j w_j * dP2_j/dr_k = sum_j w_j * 2*P2_j * d(log_P2_j)/dr_k
    # d(log_P2_j)/dr_k = -4 / (x_j - r_k)
    # dB/dr_k = sum_j w_j * P_j * (-2 / (x_j - r_k))
    # dC4/dr_k = (dA/dr_k * B^2 - A * 2*B * dB/dr_k) / B^4
    #          = (dA/dr_k) / B^2 - 2*A * dB/dr_k / B^3
    #          = (dA/dr_k) / B^2 - 2*C4 * dB/dr_k / B
    
    # Compute d(log_P)/dr_k = -2 / (x_i - r_k) for each i,k
    # inv_diff[i,k] = 1/(x_i - r_k)
    inv_diff = 1.0 / (diff + 1e-300)  # shape (n_quad, 12)
    
    # dA/dr_k = sum_i w_i * 2 * P2_i * (-4 * inv_diff[i,k])
    #          = -8 * sum_i w_i * P2_i * inv_diff[:,k]
    dA = -8.0 * (w * P2) @ inv_diff  # shape (12,)
    
    # dB/dr_k = sum_i w_i * P_i * (-2 * inv_diff[i,k])
    #          = -2 * sum_i w_i * P_i * inv_diff[:,k]
    dB = -2.0 * (w * P) @ inv_diff  # shape (12,)
    
    grad = dA / (int_P * int_P) - 2.0 * c4 * dB / int_P
    
    return c4, grad

def _objective(roots):
    """Objective function for optimization (value only)."""
    val, _ = _compute_c4_and_grad(roots)
    return val

def _objective_grad(roots):
    """Objective with gradient for L-BFGS-B."""
    return _compute_c4_and_grad(roots)

def _roots_from_tridiagonal_params(params):
    """
    Reconstruct 12 positive distinct roots from tridiagonal matrix parameters.
    params = [d_0, ..., d_11, e_0, ..., e_10] (diagonals then off-diagonals)
    The eigenvalues of the symmetric tridiagonal matrix are the roots.
    We shift to ensure positivity by adding a large constant.
    """
    d = params[:12]
    e = params[12:]  # 11 off-diagonal entries
    
    # Build tridiagonal matrix
    T = np.zeros((12, 12))
    for i in range(12):
        T[i, i] = d[i]
    for i in range(11):
        T[i, i+1] = e[i]
        T[i+1, i] = e[i]
    
    # Eigenvalues (real since symmetric)
    eigvals = np.linalg.eigvalsh(T)
    eigvals = np.sort(eigvals)
    
    # Shift to ensure positivity: find min, add offset
    min_val = eigvals[0]
    if min_val <= 0:
        eigvals = eigvals - min_val + 0.1
    
    return eigvals

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the C4 upper bound.
    
    Strategy:
    1. Use standard Laguerre roots as a strong initial guess.
    2. Generate diverse initial points.
    3. Refine with L-BFGS-B using analytical gradients.
    4. Final high-precision polish.
    """
    start_time = time.time()
    deadline = start_time + 575.0  # Safety margin before 630s kill
    
    # Initial guess: Zeros of Laguerre L_12(x)
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score, _ = _compute_c4_and_grad(best_roots)
    
    if time.time() > deadline - 120:
        return [float(xi) for xi in best_roots]
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points
    initial_points = [x0.copy()]
    
    # 1. Perturbed Laguerre roots with different scales
    for scale in [0.03, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5]:
        perturbed = x0 * (1 + scale * rng.standard_normal(12))
        perturbed = np.maximum(perturbed, 1e-3)
        perturbed = np.sort(perturbed)
        initial_points.append(perturbed)
        
    # 2. Scaled Laguerre roots
    for scale in [0.7, 0.8, 0.9, 1.1, 1.2, 1.4, 1.6, 2.0]:
        scaled = x0 * scale
        initial_points.append(scaled.copy())
        
    # 3. Random uniform in [0.1, 40]
    for _ in range(8):
        rand_roots = rng.uniform(0.1, 40, size=12)
        rand_roots = np.sort(rand_roots)
        initial_points.append(rand_roots)
        
    # 4. Geometric spacing variations
    for start in [0.3, 0.5, 1.0, 1.5, 2.0]:
        for ratio in [1.2, 1.3, 1.4, 1.5, 1.6, 1.8]:
            geom = start * (ratio ** np.arange(12))
            if geom[-1] < 150 and geom[0] > 0.05:
                initial_points.append(geom.copy())
                
    # 5. Log-uniform spacing
    for log_start in [-3.0, -2.0, -1.5, -1.0, -0.5]:
        for log_end in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            if log_end > log_start:
                log_roots = np.linspace(log_start, log_end, 12)
                roots = np.exp(log_roots)
                if roots[0] > 1e-3 and roots[-1] < 150:
                    initial_points.append(roots)
    
    # 6. Tridiagonal-based initial points
    # Start from Laguerre roots, decompose to tridiagonal, perturb
    for perturb_scale in [0.05, 0.1, 0.2]:
        # Get tridiagonal representation of diag(x0)
        d_init = x0.copy()
        e_init = np.zeros(11)
        # Add small off-diagonals to create smooth variations
        e_init = perturb_scale * np.ones(11) * rng.uniform(0.5, 1.5, size=11)
        params = np.concatenate([d_init, e_init])
        roots_from_tri = _roots_from_tridiagonal_params(params)
        if np.all(roots_from_tri > 0):
            initial_points.append(roots_from_tri)
    
    # Refine each initial point with L-BFGS-B using analytical gradient
    bounds = [(1e-6, None)] * 12
    
    n_initial = len(initial_points)
    for i, x_init in enumerate(initial_points):
        if time.time() > deadline - 30:
            break
        
        try:
            # Use L-BFGS-B with analytical gradient
            result = minimize(
                _objective,
                x_init,
                jac=_objective_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            candidate = result.x
            if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                candidate = np.sort(candidate)
                score, _ = _compute_c4_and_grad(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass
    
    # Final polish on the best roots with tighter tolerance and more iterations
    if time.time() < deadline - 5:
        try:
            result_final = minimize(
                _objective,
                best_roots,
                jac=_objective_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 100}
            )
            if np.all(result_final.x > 0) and not np.any(np.isnan(result_final.x)):
                candidate = np.sort(result_final.x)
                score, _ = _compute_c4_and_grad(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass
    
    # Ensure the output is a list of 12 positive floats
    result = [max(float(r), 1e-6) for r in best_roots]
    if len(result) != 12:
        result = [float(xi) for xi in x0]
        
    return result
# EVOLVE-BLOCK-END
