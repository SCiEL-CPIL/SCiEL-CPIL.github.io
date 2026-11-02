import numpy as np
from scipy.optimize import minimize
import time

N = 300

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # 1. Initialization: Hexagonal Lattice
    # Reuse logic from seed to get a good start
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            p0 = points[np.argsort(radius, kind="stable")[:N]].copy()
            break
        spacing *= 0.98
        
    # Ensure p0 is shape (N, 2)
    if p0.shape[0] < N:
         # Fallback random if lattice fails
         p0 = np.random.uniform(-0.9, 0.9, (N, 2))

    # 2. Objective and Gradient Functions
    def objective_and_grad(p_flat, beta):
        p = p_flat.reshape(N, 2)
        
        # Calculate pairwise distances
        # diff[i, j] = p[i] - p[j]
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Mask diagonal (self-distance)
        np.fill_diagonal(dist, np.inf)
        
        # Clearances
        # Boundary clearances
        r_norm = np.linalg.norm(p, axis=1)
        c_bound = 1.0 - r_norm
        
        # Pair clearances
        c_pair = dist / 2.0
        
        # Flatten pair clearances to 1D array for log-sum-exp
        # We only need upper triangle to avoid double counting, 
        # but for gradient accumulation, symmetry helps.
        # Let's just use all i != j, but divide the final gradient contribution by 2? 
        # No, c_ij = c_ji. If we sum over all i != j, we count each pair twice.
        # The min is the same. The soft min of {x, x, y} is slightly different from {x, y}.
        # To be consistent, let's use upper triangle indices.
        i_idx, j_idx = np.triu_indices(N, k=1)
        c_pairs = c_pair[i_idx, j_idx]
        
        all_c = np.concatenate([c_bound, c_pairs])
        
        # Soft min: -1/beta * log(sum(exp(-beta * c)))
        # To avoid overflow, shift by max
        max_c = np.max(all_c)
        exp_terms = np.exp(-beta * (all_c - max_c))
        sum_exp = np.sum(exp_terms)
        log_sum_exp = np.log(sum_exp) + (-beta * max_c)
        
        obj = -log_sum_exp / beta
        
        # Gradient
        # dF/dc_k = - exp(-beta c_k) / sum_exp
        weights = -exp_terms / sum_exp
        
        grad = np.zeros_like(p)
        
        # Boundary gradient contribution
        # c_i = 1 - |p_i|
        # d c_i / d p_i = - p_i / |p_i|
        # d F / d p_i += w_i * (- p_i / |p_i|)
        # Note: if |p_i| == 0, gradient is 0 (or undefined, treat as 0)
        r_norm_safe = np.where(r_norm > 1e-8, r_norm, 1.0)
        grad_bound = -p / r_norm_safe[:, np.newaxis]
        grad_bound *= np.where(r_norm > 1e-8, 1.0, 0.0)[:, np.newaxis]
        
        grad += (weights[:N] * grad_bound)
        
        # Pair gradient contribution
        # c_ij = 0.5 * |p_i - p_j|
        # d c_ij / d p_i = 0.5 * (p_i - p_j) / |p_i - p_j|
        # d c_ij / d p_j = 0.5 * (p_j - p_i) / |p_i - p_j|
        
        # We have weights for each pair in c_pairs
        # w_k corresponds to pair (i_idx[k], j_idx[k])
        
        # Vectorized update
        # diff_ij = p[i] - p[j]
        # dist_ij = |diff_ij|
        # unit_vec = diff_ij / dist_ij
        
        diff_pairs = p[i_idx] - p[j_idx]
        dist_pairs = c_pairs * 2.0 # This is |p_i - p_j|
        
        # Avoid division by zero
        dist_pairs_safe = np.where(dist_pairs > 1e-8, dist_pairs, 1.0)
        unit_vecs = diff_pairs / dist_pairs_safe[:, np.newaxis]
        
        # Contribution to p_i: + w_k * 0.5 * unit_vec
        # Contribution to p_j: - w_k * 0.5 * unit_vec
        
        # Scatter add
        # np.add.at(grad, i_idx, 0.5 * weights[N:] * unit_vecs)
        # np.add.at(grad, j_idx, -0.5 * weights[N:] * unit_vecs)
        
        # This is slow for N=300? 45k pairs. add.at is okay.
        # But we can do it with bincount or similar? No, vector of vectors.
        # Let's use add.at.
        
        w_pairs = weights[N:] * 0.5
        
        np.add.at(grad, i_idx, w_pairs * unit_vecs)
        np.add.at(grad, j_idx, -w_pairs * unit_vecs)
        
        return obj, grad

    # 3. Optimization Loop
    best_p = p0
    best_score = -np.inf
    
    # Evaluate initial score (hard min)
    def calc_score(p):
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist) / 2.0
        min_bound = np.min(1.0 - np.linalg.norm(p, axis=1))
        return min(min_pair, min_bound)
    
    current_score = calc_score(p0)
    if current_score > best_score:
        best_score = current_score
        best_p = p0.copy()

    betas = [5, 20, 50, 100, 200, 500]
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
            
        p_flat = best_p.flatten()
        
        res = minimize(
            lambda x: objective_and_grad(x, beta)[0],
            p_flat,
            jac=lambda x: objective_and_grad(x, beta)[1],
            method='L-BFGS-B',
            bounds=[(-1.0, 1.0)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if res.success or res.status == 0:
            new_p = res.x.reshape(N, 2)
            new_score = calc_score(new_p)
            if new_score > best_score:
                best_score = new_score
                best_p = new_p.copy()
                
    return best_p
