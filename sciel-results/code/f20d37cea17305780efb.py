"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _soft_min_radius(C, beta=50.0):
    """
    Smooth surrogate for the packing radius using log-sum-exp.
    r = min(min_{i<j} |c_i - c_j|/2, min_i min_k (1/2 - |c_ik|))
    We maximize r, so we minimize -r.
    """
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    
    # Pairwise distances / 2
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    pair_terms = dists / 2.0
    
    # Boundary distances: 1/2 - |c_ik| for each coordinate
    abs_C = np.abs(C)
    boundary_terms = 0.5 - abs_C  # shape (n, 3)
    
    # Combine all constraints into one vector
    all_terms = np.concatenate([pair_terms[np.triu_indices(n, k=1)], 
                                 boundary_terms.flatten()])
    
    # Soft min: (1/beta) * log(sum(exp(-beta * x))) approximates min(x) for large beta
    min_val = np.min(all_terms)
    shifted = -beta * (all_terms - min_val)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min_val = min_val - (1.0 / beta) * log_sum
    
    return -soft_min_val  # We minimize this (maximize radius)


def _grad_soft_min_radius(C, beta=50.0):
    """Gradient of the soft-min radius surrogate, vectorized."""
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-30)
    np.fill_diagonal(dists, 1.0)
    pair_terms = dists / 2.0
    
    abs_C = np.abs(C)
    boundary_terms = 0.5 - abs_C
    
    all_terms = np.concatenate([pair_terms[np.triu_indices(n, k=1)], 
                                 boundary_terms.flatten()])
    
    # Soft min weights
    min_val = np.min(all_terms)
    shifted = -beta * (all_terms - min_val)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    grad_C = np.zeros_like(C)
    
    # Pairwise part - vectorized
    n_pairs = n * (n - 1) // 2
    pair_grad = weights[:n_pairs]
    triu_i, triu_j = np.triu_indices(n, k=1)
    
    # d(term)/d(C_i) = (C_i - C_j) / (2 * dist) * weight
    # d(term)/d(C_j) = -(C_i - C_j) / (2 * dist) * weight
    d = C[triu_i] - C[triu_j]  # shape (n_pairs, 3)
    dist = np.sqrt(np.sum(d ** 2, axis=1) + 1e-30)  # shape (n_pairs,)
    
    # Avoid division by zero
    valid = dist > 1e-15
    coeff = np.zeros(n_pairs)
    coeff[valid] = pair_grad[valid] / (2.0 * dist[valid])
    
    # Accumulate gradients using np.add.at for scattered updates
    weighted_d = coeff[:, np.newaxis] * d  # shape (n_pairs, 3)
    
    np.add.at(grad_C, triu_i, weighted_d)
    np.add.at(grad_C, triu_j, -weighted_d)
    
    # Boundary part
    bnd_grad = weights[n_pairs:]
    bnd_grad_reshaped = bnd_grad.reshape(n, 3)
    signs = np.sign(C)
    signs[abs_C < 1e-15] = 1.0
    grad_C += -bnd_grad_reshaped * signs
    
    return grad_C


def _exact_radius(C):
    """Compute the exact packing radius."""
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    if n < 2:
        return np.inf
    
    diff = C[:, np.newaxis, :] - C[np.newaxis, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    abs_C = np.abs(C)
    min_bnd = np.min(0.5 - abs_C)
    
    return min(min_pair, min_bnd)


def _initial_placement():
    """Generate a better initial placement using a jittered grid."""
    m = 4  # 4x4x4 = 64 positions, take 50
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Add small random jitter to break symmetry
    rng = np.random.default_rng(42)
    C = grid[:N].copy()
    C += rng.uniform(-0.02, 0.02, size=C.shape)
    
    # Clip to ensure valid
    C = np.clip(C, -0.48, 0.48)
    
    return C


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    C0 = _initial_placement()
    
    best_C = C0.copy()
    best_r = _exact_radius(C0)
    
    # Try multiple betas (from smooth to sharp)
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    bounds = [(-0.4999, 0.4999)] * (N * 3)
    
    for beta in betas:
        if time.time() > deadline - 25.0:
            break
        
        def objective(x):
            C = x.reshape(-1, 3)
            return _soft_min_radius(C, beta)
        
        def gradient(x):
            C = x.reshape(-1, 3)
            g = _grad_soft_min_radius(C, beta)
            return g.flatten()
        
        x0 = best_C.flatten()
        
        try:
            res = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            C_new = res.x.reshape(-1, 3)
            # Clip to valid range
            C_new = np.clip(C_new, -0.4999, 0.4999)
            r_new = _exact_radius(C_new)
            
            if r_new > best_r:
                best_r = r_new
                best_C = C_new.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 15.0:
        beta_final = 1000.0
        
        def objective_final(x):
            C = x.reshape(-1, 3)
            return _soft_min_radius(C, beta_final)
        
        def gradient_final(x):
            C = x.reshape(-1, 3)
            g = _grad_soft_min_radius(C, beta_final)
            return g.flatten()
        
        try:
            res = minimize(
                objective_final,
                best_C.flatten(),
                jac=gradient_final,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 100}
            )
            C_new = res.x.reshape(-1, 3)
            C_new = np.clip(C_new, -0.4999, 0.4999)
            r_new = _exact_radius(C_new)
            if r_new > best_r:
                best_r = r_new
                best_C = C_new.copy()
        except Exception:
            pass
    
    # Ensure validity: clip to cube and check
    best_C = np.clip(best_C, -0.5 + 1e-15, 0.5 - 1e-15)
    
    return best_C
# EVOLVE-BLOCK-END
