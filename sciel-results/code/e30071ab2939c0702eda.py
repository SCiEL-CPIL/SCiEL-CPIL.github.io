"""
Improved packer: Soft-min surrogate optimization with L-BFGS-B.
Combines the robust grid initialization of the current program with the 
smooth soft-min objective from the inspiration, fully vectorized for speed.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100

def _score(points):
    """Largest common radius: min of half pairwise distances and wall clearance."""
    # Vectorized distance calculation
    d = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(d * d, axis=2))
    iu = np.triu_indices(N, k=1)
    min_pair = dists[iu].min()
    clearance = (0.5 - np.abs(points)).min()
    return min(min_pair * 0.5, clearance)

def _soft_min_obj_grad(x, temperature):
    """
    Compute objective and gradient for the soft-min of distances.
    We minimize -soft_min(distances) = temperature * log(sum(exp(-d_i/T)))
    where d_i are the pairwise distances/2 and wall distances.
    
    x: (N*3,) array of centers
    temperature: float, controls the sharpness of the min
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    # 1. Pairwise distances
    # diff[i, j] = centers[i] - centers[j]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    
    # We only care about i < j
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists_sq = dists_sq[i_idx, j_idx]
    
    # Avoid division by zero in gradients if points coincide
    # Add a tiny epsilon to squared distances before sqrt
    pair_dists = np.sqrt(pair_dists_sq + 1e-15)
    # The constraint is distance >= 2r, so the "distance metric" for the min is d/2
    # Let a_i = d_ij / 2
    a_pairs = pair_dists * 0.5
    
    # 2. Wall distances
    # Wall distance for point i is min(0.5 - |c_ix|, 0.5 - |c_iy|, 0.5 - |c_iz|)
    # This is non-smooth. To make it smooth for the soft-min, we can use a similar trick
    # or just include the 3 wall distances per point as separate terms in the soft-min.
    # Including all 3 wall terms per point (left, right, front, back, etc.) is standard.
    # Wall terms: 0.5 - c_ik and 0.5 + c_ik for k=0,1,2.
    
    # Let's construct the full vector of 'distances' a_k
    # a_k are the values we want to maximize the minimum of.
    
    # Wall terms:
    # For each point i and coord k:
    # d_wall1 = 0.5 - centers[i, k]
    # d_wall2 = 0.5 + centers[i, k]
    
    # To be efficient, we can compute gradients analytically.
    # Objective: f(x) = T * log( sum_k exp(-a_k/T) )
    # Gradient: grad f = sum_k [ exp(-a_k/T) / sum_j exp(-a_j/T) ] * (-grad a_k)
    # Let w_k = exp(-a_k/T) / Z, where Z = sum exp(-a_j/T)
    # grad f = - sum_k w_k * grad a_k
    
    # We need to compute weights w_k for all pairs and all walls.
    
    # --- Pairwise part ---
    # a_pair = d/2
    # grad a_pair w.r.t c_i = (c_i - c_j) / (2 * d)
    # grad a_pair w.r.t c_j = (c_j - c_i) / (2 * d) = - grad a_pair w.r.t c_i
    
    scaled_pairs = -a_pairs / temperature
    max_scaled_pairs = np.max(scaled_pairs)
    exp_pairs = np.exp(scaled_pairs - max_scaled_pairs)
    
    # --- Wall part ---
    # There are 3 coords * 2 sides * N points = 6N wall terms.
    # Let's create arrays for them.
    # w_terms[i, k, side] where side 0 is (0.5 - c), side 1 is (0.5 + c)
    # d_0 = 0.5 - c_ik
    # d_1 = 0.5 + c_ik
    
    c = centers
    wall_d0 = 0.5 - c  # (N, 3)
    wall_d1 = 0.5 + c  # (N, 3)
    
    # Flatten to 1D for uniform handling? Or keep structured.
    # Let's keep structured for gradient accumulation.
    
    scaled_walls0 = -wall_d0 / temperature
    scaled_walls1 = -wall_d1 / temperature
    
    max_scaled_walls = np.max(np.maximum(scaled_walls0, scaled_walls1))
    
    # To avoid overflow in exp, we should normalize globally.
    # Let M = max(max_scaled_pairs, max_scaled_walls)
    M = np.maximum(max_scaled_pairs, max_scaled_walls)
    
    exp_pairs_norm = np.exp(scaled_pairs - M)
    exp_walls0_norm = np.exp(scaled_walls0 - M)
    exp_walls1_norm = np.exp(scaled_walls1 - M)
    
    sum_exp = np.sum(exp_pairs_norm) + np.sum(exp_walls0_norm) + np.sum(exp_walls1_norm)
    
    if sum_exp < 1e-300:
        # Fallback, should not happen
        return 0.0, np.zeros_like(x)
    
    # Weights
    w_pairs = exp_pairs_norm / sum_exp
    w_walls0 = exp_walls0_norm / sum_exp
    w_walls1 = exp_walls1_norm / sum_exp
    
    # Objective value
    # f = T * (log(sum_exp) + M)
    obj = temperature * (np.log(sum_exp) + M)
    
    # Gradient
    # grad = - sum w_k * grad a_k
    grad = np.zeros_like(centers)
    
    # 1. Pairwise contributions
    # For each pair (i, j) with weight w:
    # grad a_pair = (c_i - c_j) / (2 * d)
    # Contribution to grad c_i: -w * (c_i - c_j) / (2 * d)
    # Contribution to grad c_j: -w * (c_j - c_i) / (2 * d) = w * (c_i - c_j) / (2 * d)
    
    # Vectorized pairwise gradient
    # We have w_pairs (length num_pairs)
    # We need to accumulate into grad (N, 3)
    
    # Create a zero matrix for pair gradients
    # Actually, we can use np.add.at or similar, but let's try to vectorize.
    
    # diff[i, j] is available. 
    # unit_vecs = diff / dists[..., None]
    # But we only have upper triangle.
    
    # Let's compute the vector (c_i - c_j) for all pairs
    pair_vecs = diff[i_idx, j_idx, :] # (num_pairs, 3)
    pair_dists_full = pair_dists # (num_pairs,)
    
    # Gradient of a_pair = d/2 is (c_i - c_j) / (2 * d)
    # So -w * grad a = -w * (c_i - c_j) / (2 * d)
    
    # Factor = w_pairs / (2 * pair_dists)
    # Handle zero distances
    safe_dists = np.where(pair_dists_full > 1e-10, pair_dists_full, 1.0)
    factors = w_pairs / (2.0 * safe_dists)
    
    # grad_contrib = factors[:, None] * pair_vecs
    grad_contrib = factors[:, None] * pair_vecs
    
    # Accumulate into grad
    # grad[i] -= grad_contrib
    # grad[j] += grad_contrib
    
    # Using np.add.at
    np.add.at(grad, i_idx, -grad_contrib)
    np.add.at(grad, j_idx, grad_contrib)
    
    # 2. Wall contributions
    # a_wall0 = 0.5 - c_ik
    # grad a_wall0 = -e_k (unit vector in coord k)
    # Contribution: -w * grad a = -w * (-e_k) = w * e_k
    # So grad c_ik += w_walls0[i, k]
    
    # a_wall1 = 0.5 + c_ik
    # grad a_wall1 = e_k
    # Contribution: -w * grad a = -w * e_k
    # So grad c_ik -= w_walls1[i, k]
    
    grad += w_walls0
    grad -= w_walls1
    
    return obj, grad.flatten()

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # 10s safety margin under 150s
    
    # ---- Seed: simple cubic 5x5x5 grid ----
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pts = grid[:N].copy()
    
    best_pts = pts.copy()
    best_r = _score(best_pts)
    
    rng = np.random.RandomState(42)
    
    # Bounds for L-BFGS-B
    # Centers must be within [-0.5, 0.5]
    # To avoid boundary issues with gradients, we can set bounds slightly inside
    # or let the optimizer handle it. The soft-min penalty pushes them in.
    # Setting bounds to [-0.5, 0.5] is correct.
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Annealing schedule for temperature
    # Start high to explore, end low to refine
    temperatures = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    current_pts = pts.copy()
    
    for temp in temperatures:
        if time.time() - t0 > DEADLINE - 10.0:
            break
            
        def objective(x):
            return _soft_min_obj_grad(x, temp)
            
        try:
            result = minimize(
                objective,
                current_pts.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-12, 'disp': False, 'maxls': 50}
            )
            opt_pts = result.x.reshape(-1, 3)
            
            # Validate and update
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -0.5, 0.5)
                r_opt = _score(opt_pts)
                if r_opt > best_r:
                    best_r = r_opt
                    best_pts = opt_pts.copy()
                
                # Update current for next iteration
                # Only move if it improved or if we are in refinement phase?
                # Usually, we follow the optimizer's path.
                current_pts = opt_pts
                
        except Exception as e:
            # If optimizer fails, keep current
            pass

    # Final Polish with very low temperature
    if time.time() - t0 < DEADLINE - 5.0:
        final_temp = 1e-5
        def final_obj(x):
            return _soft_min_obj_grad(x, final_temp)
            
        try:
            result = minimize(
                final_obj,
                best_pts.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14, 'disp': False, 'maxls': 50}
            )
            opt_pts = result.x.reshape(-1, 3)
            if np.all(np.isfinite(opt_pts)):
                opt_pts = np.clip(opt_pts, -0.5, 0.5)
                r_opt = _score(opt_pts)
                if r_opt > best_r:
                    best_r = r_opt
                    best_pts = opt_pts.copy()
        except Exception:
            pass

    # Final safety
    best_pts = np.clip(best_pts, -0.5, 0.5)
    if not np.all(np.isfinite(best_pts)):
        best_pts = grid[:N].copy()
        
    return best_pts
