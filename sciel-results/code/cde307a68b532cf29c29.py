"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Precompute indices for pairwise distances
    idx_i, idx_j = np.triu_indices(N, 1)
    num_pairs = len(idx_i)
    
    def compute_radius(x):
        """Compute the exact maximum radius for given positions."""
        positions = x.reshape(-1, 3)
        # Minimum distance to boundary
        boundary_dists = 0.5 - np.abs(positions)
        min_boundary = np.min(boundary_dists)
        
        # Minimum pairwise distance / 2
        if N > 1:
            min_pair = np.min(pdist(positions)) / 2.0
        else:
            min_pair = 0.5
            
        return min(min_boundary, min_pair)
    
    def negative_soft_radius(x, beta):
        """Smooth objective: negative of soft-min radius."""
        positions = x.reshape(-1, 3)
        # Boundary distances
        boundary_dists = 0.5 - np.abs(positions)
        # Pairwise distances / 2
        diffs = positions[idx_i] - positions[idx_j]
        pair_dists = np.linalg.norm(diffs, axis=1) / 2.0
        
        # Combine all constraints into a single soft-min
        # We want to maximize min(boundary_dists, pair_dists)
        # This is equivalent to minimizing -min(...)
        # Soft min of values v is -logsumexp(-v/beta) * beta
        # We apply this to the combined vector of all distances
        all_dists = np.concatenate([boundary_dists, pair_dists])
        
        # Clip to avoid numerical issues
        all_dists = np.clip(all_dists, -1e6, 1e6)
        
        # Soft min
        neg_vals = -all_dists / beta
        max_neg = np.max(neg_vals)
        log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_vals - max_neg)))
        soft_min_val = -log_sum_exp * beta
        
        return -soft_min_val
    
    def negative_soft_radius_grad(x, beta):
        """Gradient of the smooth objective."""
        positions = x.reshape(-1, 3)
        boundary_dists = 0.5 - np.abs(positions)
        diffs = positions[idx_i] - positions[idx_j]
        dists = np.linalg.norm(diffs, axis=1)
        dists_safe = np.maximum(dists, 1e-12)
        pair_dists = dists_safe / 2.0
        
        all_dists = np.concatenate([boundary_dists, pair_dists])
        all_dists = np.clip(all_dists, -1e6, 1e6)
        
        # Weights for soft min
        neg_vals = -all_dists / beta
        max_neg = np.max(neg_vals)
        exp_vals = np.exp(neg_vals - max_neg)
        weights = exp_vals / np.sum(exp_vals)
        
        # Gradient of -soft_min w.r.t. distances is -weights
        grad_dists = -weights
        
        grad = np.zeros_like(x)
        
        # Boundary part
        # d(boundary_dist_k)/dx_k = -sign(x_k)
        # grad_dists[0:N] corresponds to boundary distances
        w_boundary = grad_dists[:N]
        sign_x = np.sign(positions)
        # grad_boundary = -sign_x * w_boundary
        grad_boundary = -sign_x * w_boundary.reshape(-1, 3)
        grad[:N*3] += grad_boundary.flatten()
        
        # Pairwise part
        # d(pair_dist_k)/dx_i = (x_i - x_j) / (2 * ||x_i - x_j||)
        # d(pair_dist_k)/dx_j = -(x_i - x_j) / (2 * ||x_i - x_j||)
        w_pairs = grad_dists[N:]
        
        unit_vecs = diffs / dists_safe[:, None]
        
        # Accumulate gradients for i and j
        # grad_i += sum_k (w_k * unit_vec_k / 2)
        # grad_j -= sum_k (w_k * unit_vec_k / 2)
        
        # Use numpy bincount for efficiency instead of loop
        # But since we need to update specific indices, a loop or scatter is needed.
        # For N=50, num_pairs=1225. Loop is fast enough.
        for k in range(num_pairs):
            i = idx_i[k]
            j = idx_j[k]
            w = w_pairs[k]
            vec = unit_vecs[k] * 0.5 * w
            grad[3*i:3*i+3] += vec
            grad[3*j:3*j+3] -= vec
            
        return grad

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_x = None
    best_r = -1.0
    
    rng = np.random.RandomState(42)
    
    # List of initializations
    inits = []
    
    # 1. Grid initialization
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    inits.append(grid[:N].flatten())
    
    # 2. Random initializations
    for _ in range(10):
        x_rand = rng.uniform(-0.4, 0.4, (N, 3))
        inits.append(x_rand.flatten())
        
    # 3. Jittered grid
    for _ in range(5):
        x_jitter = grid[:N] + rng.randn(N, 3) * 0.1
        x_jitter = np.clip(x_jitter, -0.45, 0.45)
        inits.append(x_jitter.flatten())
        
    # Optimization strategy:
    # For each init, run optimization with increasing beta.
    # Beta schedule: [10, 20, 50, 100]
    betas = [10.0, 20.0, 50.0, 100.0]
    
    num_restarts = len(inits)
    
    for i_init, x0 in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
            
        x_current = x0.copy()
        
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
                
            try:
                result = minimize(
                    negative_soft_radius, 
                    x_current, 
                    method='L-BFGS-B',
                    bounds=bounds,
                    jac=lambda x: negative_soft_radius_grad(x, beta),
                    args=(beta,),
                    options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10}
                )
                
                if result.success:
                    x_current = result.x
            except Exception:
                # If it fails, keep x_current
                pass
        
        # Evaluate final radius for this restart
        r_current = compute_radius(x_current)
        if r_current > best_r:
            best_r = r_current
            best_x = x_current.copy()
            
    # Final polish with exact objective and derivative-free method
    if best_x is not None and time.time() < deadline - 5.0:
        try:
            def neg_radius_exact(x):
                return -compute_radius(x)
            
            # Use Nelder-Mead for local refinement without gradients
            # It handles non-smooth functions well
            result = minimize(
                neg_radius_exact, 
                best_x, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            r_final = compute_radius(result.x)
            if r_final > best_r:
                best_r = r_final
                best_x = result.x.copy()
        except Exception:
            pass
            
    # Ensure bounds are respected strictly
    if best_x is not None:
        best_x = np.clip(best_x, -0.5, 0.5)
        
    return best_x.reshape(-1, 3)
# EVOLVE-BLOCK-END
