"""Rewritten packing solver: multi-start gradient ascent on the minimum-pair-distance objective."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
HALF = 0.5
EPS = 1e-12


def _score(c):
    """Compute the packing score (minimum radius) for centre array c of shape (N,3)."""
    # Boundary constraint: distance to nearest face
    bound = HALF - np.max(np.abs(c), axis=1)
    r_bound = np.min(bound)
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    r_pair = 0.5 * np.min(d)
    return min(r_bound, r_pair)


def _soft_min_obj_and_grad(c, tau):
    """
    Compute the negative soft-min of the radii and its gradient.
    We want to maximize soft_min(R), so we minimize -soft_min(R).
    
    R consists of:
    - Boundary radii: b_i = 0.5 - ||c_i||_inf
    - Pair radii: p_ij = 0.5 * ||c_i - c_j||
    
    soft_min(R, tau) = -tau * log(sum(exp(-R_k / tau)))
    grad_c soft_min = - sum_k w_k * grad_c R_k
    where w_k = exp(-R_k / tau) / sum(exp(-R_m / tau))
    
    We return (-soft_min, -grad_soft_min) to minimize.
    """
    # 1. Compute Boundary Radii
    # b_i = 0.5 - max(|c_ix|, |c_iy|, |c_iz|)
    abs_c = np.abs(c)
    max_abs = np.max(abs_c, axis=1)
    b = HALF - max_abs
    
    # 2. Compute Pair Radii
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    p = 0.5 * d
    
    # 3. Assemble all radii
    # We need to be careful with memory. N=100, so 4950 pairs + 100 bounds = 5050 values.
    # This is small.
    all_r = np.concatenate([b, p[np.triu_indices(N, 1)]])
    
    # 4. Soft Min
    # To avoid overflow/underflow, we can shift, but tau is small, so exp(-R/tau) can underflow for large R.
    # Actually, we want exp(-R/tau). If R is large, exp is 0.
    # Let's use log-sum-exp trick for stability if needed, but for N=100 it's likely fine.
    # However, if tau is very small, exp(-R/tau) becomes 0 for most R, effectively picking the min.
    
    # Calculate weights w_k
    # w_k = exp(-R_k / tau) / Z
    # To be stable:
    # max_R = max(all_r)
    # exp_terms = exp(-(all_r - max_R) / tau)
    # Z = sum(exp_terms)
    # w = exp_terms / Z
    # soft_min_val = max_R - tau * log(Z)
    
    max_R = np.max(all_r)
    exp_terms = np.exp(-(all_r - max_R) / tau)
    Z = np.sum(exp_terms)
    w = exp_terms / Z
    
    soft_min_val = max_R - tau * np.log(Z)
    
    # 5. Gradient
    # grad_c soft_min = - sum_k w_k * grad_c R_k
    # We need to compute sum_k w_k * grad_c R_k
    
    grad = np.zeros_like(c)
    
    # Unpack weights
    w_bounds = w[:N]
    w_pairs = w[N:]
    
    # Process Boundary Weights
    # R_k = b_i = 0.5 - max(|c_i|)
    # grad_c b_i = - sign(c_i) * 1_{active_dim}
    # For each i, if w_bounds[i] > 0:
    #   grad[i] += -w_bounds[i] * grad_b_i
    
    # Vectorized approach for boundaries:
    # For each dimension k in {0,1,2}, the contribution to grad[i,k] from b_i is:
    # -w_bounds[i] * sign(c_i,k) * 1_{|c_i,k| == max_abs_i}
    # We can compute this for all i and k.
    
    # active_mask[i, k] = 1 if |c_i,k| is close to max_abs_i
    # Due to floating point, use tolerance
    tol = 1e-9
    active_mask = (np.abs(abs_c - max_abs[:, None]) < tol).astype(float)
    
    # grad_b_i_k = -sign(c_i,k) * active_mask[i,k]
    grad_b = -np.sign(c) * active_mask
    
    # Weighted sum for boundaries
    # grad[i, k] += sum over active boundaries? No, each boundary b_i corresponds to one weight w_bounds[i].
    # So grad[i, k] += w_bounds[i] * grad_b[i, k]
    grad += w_bounds[:, None] * grad_b
    
    # Process Pair Weights
    # R_k = p_ij = 0.5 * ||c_i - c_j||
    # grad_c_i p_ij = 0.5 * (c_i - c_j) / ||c_i - c_j||
    # grad_c_j p_ij = -0.5 * (c_i - c_j) / ||c_i - c_j||
    
    # w_pairs is flattened from upper triangle.
    # We need to map weights back to (i, j) pairs.
    i_idx, j_idx = np.triu_indices(N, 1)
    
    # diff[i, j] = c[i] - c[j]
    # d[i, j] = ||diff[i, j]||
    # unit[i, j] = diff[i, j] / d[i, j]
    
    # We only computed d for the full matrix.
    # Let's extract the upper triangle values for unit vectors.
    # d[i_idx, j_idx] are the distances.
    # diff[i_idx, j_idx] are the vectors.
    
    d_pairs = d[i_idx, j_idx]
    diff_pairs = diff[i_idx, j_idx]
    
    # Handle zero distance
    safe_d = np.where(d_pairs < EPS, 1.0, d_pairs)
    unit_pairs = diff_pairs / safe_d[:, None]
    
    # grad_c_i contribution: 0.5 * unit_pairs * w_pairs
    # grad_c_j contribution: -0.5 * unit_pairs * w_pairs
    
    # We need to scatter these into grad.
    # grad[i] += 0.5 * unit_pairs * w_pairs
    # grad[j] -= 0.5 * unit_pairs * w_pairs
    
    # Use np.add.at for accumulation
    np.add.at(grad, i_idx, 0.5 * w_pairs[:, None] * unit_pairs)
    np.add.at(grad, j_idx, -0.5 * w_pairs[:, None] * unit_pairs)
    
    # Total gradient of soft_min
    grad_soft_min = -grad
    
    # Objective is -soft_min
    obj = -soft_min_val
    grad_obj = -grad_soft_min
    
    return obj, grad_obj


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_score = -np.inf
    best_c = None
    
    # Initial guess: simple grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_c = grid[:N].copy()
    
    # Try multiple random initializations
    n_restarts = 10
    
    # Schedule of tau values (temperature for soft-min)
    # Start with larger tau for smoothness, decrease for precision
    tau_schedule = [0.05, 0.02, 0.01, 0.005, 0.001]
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            c = init_c.copy()
        else:
            # Random perturbation of the grid
            c = init_c + 0.05 * np.random.randn(N, 3)
            # Or fully random
            if restart % 2 == 0:
                c = np.random.uniform(-0.4, 0.4, (N, 3))
        
        # Ensure points are inside the cube
        c = np.clip(c, -HALF + 0.01, HALF - 0.01)
        
        # Optimize with decreasing tau
        for tau in tau_schedule:
            if time.time() > deadline:
                break
            
            try:
                result = minimize(
                    _soft_min_obj_and_grad,
                    c,
                    args=(tau,),
                    method='L-BFGS-B',
                    bounds=[(-HALF + 1e-9, HALF - 1e-9)] * N * 3,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                c_opt = result.x
                c = c_opt
            except Exception:
                continue
        
        # Evaluate the true score
        score_opt = _score(c)
        
        if score_opt > best_score:
            best_score = score_opt
            best_c = c.copy()
    
    # Final polish: try a few more iterations from the best solution with very small tau
    if best_c is not None and time.time() < deadline:
        try:
            # Use a very small tau for high precision
            tau_final = 1e-5
            result = minimize(
                _soft_min_obj_and_grad,
                best_c,
                args=(tau_final,),
                method='L-BFGS-B',
                bounds=[(-HALF + 1e-9, HALF - 1e-9)] * N * 3,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            c_opt = result.x
            score_opt = _score(c_opt)
            if score_opt > best_score:
                best_score = score_opt
                best_c = c_opt.copy()
        except Exception:
            pass
    
    # Ensure validity: clip to cube
    if best_c is None:
        return init_c[:N]
    
    best_c = np.clip(best_c, -HALF, HALF)
    return best_c
