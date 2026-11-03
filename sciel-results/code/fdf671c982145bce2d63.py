import numpy as np
from scipy.optimize import minimize
import time
import math

N = 200

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def compute_radius(centers):
        """Compute the maximum common radius for given centers."""
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        if min_wall < -1e-9:
            return 0.0
        # Use efficient distance computation
        # For N=200, full cdist is 40000 ops, which is fast enough.
        # But we can be smarter if needed.
        # dist_matrix = cdist(centers, centers)
        # np.fill_diagonal(dist_matrix, np.inf)
        # min_pair = np.min(dist_matrix)
        
        # Let's stick to a simple but robust method
        # Since we call this frequently, maybe we can optimize?
        # Actually, for the soft-min, we need the matrix anyway.
        
        dist_matrix = np.zeros((N, N))
        for i in range(N):
            dist_matrix[i, i+1:] = np.linalg.norm(centers[i] - centers[i+1:], axis=1)
            dist_matrix[i+1:, i] = dist_matrix[i, i+1:]
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def soft_min(values, beta):
        """
        Compute soft-min of a set of values.
        soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
        """
        # For numerical stability
        min_val = np.min(values)
        if min_val < -1e9:
            return min_val
        exp_vals = np.exp(-beta * (values - min_val))
        log_sum = np.log(np.sum(exp_vals))
        return min_val - np.log(log_sum) / beta

    def objective_and_grad(x, beta):
        """
        Objective: minimize -soft_min(constraints)
        Constraints:
        1. Wall distances: 0.5 - |x_i|
        2. Pair distances: 0.5 * ||x_i - x_j||
        
        We want to maximize the minimum constraint.
        So we minimize -soft_min(constraints).
        
        x is flattened array of shape (2N,).
        """
        centers = x.reshape(N, 2)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        flat_walls = wall_dists.flatten()
        
        # Pair distances
        # We only need i < j
        pair_dists = []
        # This is slow in Python loop, let's vectorize if possible
        # Or just compute full matrix
        dist_matrix = np.zeros((N, N))
        for i in range(N):
            dist_matrix[i, i+1:] = np.linalg.norm(centers[i] - centers[i+1:], axis=1)
            dist_matrix[i+1:, i] = dist_matrix[i, i+1:]
        np.fill_diagonal(dist_matrix, np.inf)
        flat_pairs = dist_matrix.flatten()
        # We only care about i<j, so take upper triangle
        iu = np.triu_indices(N, k=1)
        pair_vals = dist_matrix[iu] / 2.0
        
        # Combine all constraints
        # We want soft_min of all wall_dists and pair_dists
        # Let's compute soft_min separately or combine?
        # Combining is better for gradients.
        
        # To avoid huge arrays, let's just compute the soft_min of the combined vector
        # However, the gradient requires the exp terms.
        
        # Let's define the "energy" E = -soft_min(C) where C is vector of all constraints.
        # C = [wall_dists, pair_dists]
        
        # For efficiency, let's compute the soft_min and its gradient w.r.t C.
        # d(soft_min)/dC_k = exp(-beta * C_k) / sum(exp(-beta * C_j))
        
        # Let's construct C
        # Note: wall_dists has 2N elements, pair_dists has N(N-1)/2 elements.
        
        # To save memory and time, we can compute the terms directly.
        # Let S = sum(exp(-beta * c_k))
        # term_k = exp(-beta * c_k)
        # grad_C_k = term_k / S
        
        # We need grad_x = sum_k (grad_C_k * grad_c_k/dx)
        
        # Let's compute S first.
        # This requires iterating over all constraints.
        
        # Optimization: Use the fact that we only need the gradient.
        
        # 1. Compute all constraint values
        c_walls = 0.5 - np.abs(centers) # (N, 2)
        c_pairs = dist_matrix[iu] / 2.0 # (M,) where M = N(N-1)/2
        
        # 2. Compute exp terms
        # For stability, subtract min
        min_c = np.min(np.concatenate([c_walls.flatten(), c_pairs]))
        
        # Shifted values
        cw_shift = c_walls - min_c
        cp_shift = c_pairs - min_c
        
        exp_w = np.exp(-beta * cw_shift) # (N, 2)
        exp_p = np.exp(-beta * cp_shift) # (M,)
        
        S = np.sum(exp_w) + np.sum(exp_p)
        
        # Objective value
        obj = -(min_c - np.log(S) / beta)
        
        # Gradient
        # d(obj)/dx = - d(soft_min)/dx
        # d(soft_min)/dx = sum_k (exp_k/S * d(c_k)/dx)
        
        grad = np.zeros_like(x)
        
        # Wall gradients
        # d(0.5 - |x|)/dx = -sign(x)
        # Weight: exp_w / S
        w_weights = exp_w / S # (N, 2)
        # d(c_w)/dx = -sign(x)
        grad_w = -w_weights * np.sign(centers)
        grad += grad_w.flatten()
        
        # Pair gradients
        # d(||x_i - x_j||/2)/dx_i = (x_i - x_j) / (2 * ||x_i - x_j||)
        # d(||x_i - x_j||/2)/dx_j = (x_j - x_i) / (2 * ||x_i - x_j||)
        
        # We need to map the pair indices back to x indices.
        # iu[0] contains i, iu[1] contains j.
        i_idx, j_idx = iu
        
        # We have exp_p / S as weights for each pair.
        p_weights = exp_p / S
        
        # Compute unit vectors
        # dists = dist_matrix[iu]
        # unit_vecs = (centers[i_idx] - centers[j_idx]) / dists[:, None]
        
        # Gradient contribution to x_i: w * (x_i - x_j) / (2 * dist)
        # Gradient contribution to x_j: w * (x_j - x_i) / (2 * dist)
        
        # Let's compute vectors
        diff = centers[i_idx] - centers[j_idx]
        dists = np.linalg.norm(diff, axis=1)
        
        # Avoid division by zero
        safe_dists = np.where(dists < 1e-10, 1e-10, dists)
        unit_diffs = diff / safe_dists[:, None]
        
        # Weighted unit diffs
        # w_p is (M,), unit_diffs is (M, 2)
        # contrib = w_p[:, None] * unit_diffs * 0.5
        
        contrib = p_weights[:, None] * unit_diffs * 0.5
        
        # Scatter add to grad
        # grad[i_idx] += contrib
        # grad[j_idx] -= contrib
        
        # Using np.add.at
        np.add.at(grad, i_idx * 2 + np.arange(N)[i_idx], contrib[:, 0]) # This is wrong, x is interleaved
        # x is [x0x, x0y, x1x, x1y, ...]
        
        # Let's be careful with indices.
        # x index for center k is 2*k and 2*k+1.
        
        # Let's revert to simple loop for gradient accumulation if vectorized is tricky, 
        # but N=200, M=~20k. Looping 20k times in Python is ~0.01s. Acceptable.
        
        # Actually, let's try to vectorize properly.
        # grad is (2N,)
        
        # For each pair (i, j) with weight w:
        # grad[2*i] += w * dx
        # grad[2*i+1] += w * dy
        # grad[2*j] -= w * dx
        # grad[2*j+1] -= w * dy
        
        # contrib is (M, 2) where contrib[:, 0] is dx, contrib[:, 1] is dy
        # We need to add contrib to grad at indices 2*i, 2*i+1
        # and subtract from 2*j, 2*j+1
        
        # Create indices
        idx_i_x = i_idx * 2
        idx_i_y = i_idx * 2 + 1
        idx_j_x = j_idx * 2
        idx_j_y = j_idx * 2 + 1
        
        np.add.at(grad, idx_i_x, contrib[:, 0])
        np.add.at(grad, idx_i_y, contrib[:, 1])
        np.add.at(grad, idx_j_x, -contrib[:, 0])
        np.add.at(grad, idx_j_y, -contrib[:, 1])
        
        return obj, -grad # We minimize -soft_min, so gradient is negative of d(soft_min)/dx?
        # Wait. obj = -soft_min.
        # d(obj)/dx = - d(soft_min)/dx.
        # d(soft_min)/dx = sum (w_k * dc_k/dx).
        # So grad_obj = - sum (w_k * dc_k/dx).
        # My calculation above computed sum (w_k * dc_k/dx).
        # So I should return -grad.

    def hexagonal_init(spacing=None):
        """Generate a hexagonal lattice initial guess."""
        if spacing is None:
            spacing = 0.073
        d = spacing
        y_spacing = d * math.sqrt(3) / 2
        
        # Estimate rows and cols
        cols = int(1.0 / d) + 1
        rows = int(1.0 / y_spacing) + 1
        
        pts = []
        total_height = (rows - 1) * y_spacing
        y_start = -total_height / 2
        
        for i in range(rows):
            y = y_start + i * y_spacing
            if i % 2 == 0:
                x_start = -0.5 + d/2
                for j in range(cols):
                    x = x_start + j * d
                    pts.append([x, y])
            else:
                x_start = -0.5 + d
                for j in range(cols - 1):
                    x = x_start + j * d
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) > N:
            idx = np.random.choice(len(pts), N, replace=False)
            pts = pts[idx]
        elif len(pts) < N:
            diff = N - len(pts)
            random_pts = np.random.uniform(-0.45, 0.45, (diff, 2))
            pts = np.vstack([pts, random_pts])
        
        return pts

    def run_optimization(init_centers, beta, max_iter, time_budget):
        """Run L-BFGS-B optimization."""
        x0 = init_centers.flatten()
        
        # Bounds: x in [-0.5, 0.5]
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        start = time.time()
        
        def fun(x):
            val, grad = objective_and_grad(x, beta)
            return val, grad
        
        try:
            res = minimize(fun, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                           options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        except Exception:
            return init_centers, compute_radius(init_centers)
        
        elapsed = time.time() - start
        if elapsed > time_budget:
            return init_centers, compute_radius(init_centers)
            
        x_opt = res.x.reshape(N, 2)
        # Clip to ensure validity
        x_opt = np.clip(x_opt, -0.5, 0.5)
        
        r = compute_radius(x_opt)
        return x_opt, r

    best_centers = None
    best_radius = 0.0

    # Strategy: Multi-restart with varying beta and initializations
    # Start with coarse beta, then refine
    
    # Initializations
    inits = []
    # 1. Standard hex
    inits.append(hexagonal_init(0.073))
    # 2. Slightly different hex
    inits.append(hexagonal_init(0.074))
    inits.append(hexagonal_init(0.072))
    # 3. Square grid
    m = int(math.ceil(math.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) > N:
        idx = np.random.choice(len(grid), N, replace=False)
        grid = grid[idx]
    inits.append(grid)
    
    # 4. Random
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # Optimization phases
    # Phase 1: Coarse optimization with low beta (smoother landscape)
    # Phase 2: Fine optimization with high beta (sharper, closer to true min)
    
    betas = [10, 50, 100, 200, 500]
    
    current_centers = inits[0].copy()
    
    for i_init, init_c in enumerate(inits):
        if time.time() > deadline - 20:
            break
            
        # If we already have a good result, maybe skip some inits?
        # Let's try each init with a short budget first.
        
        time_per_init = max(5.0, (deadline - time.time() - 20) / (len(inits) - i_init))
        
        # Run a quick optimization with medium beta
        polished, r = run_optimization(init_c, beta=100, max_iter=50, time_budget=time_per_init)
        
        if r > best_radius:
            best_radius = r
            best_centers = polished.copy()
            
        # If time remains, refine the best found so far
        if best_centers is not None and time.time() < deadline - 10:
             # Try to refine best_centers with higher beta
             time_left = deadline - time.time() - 5
             if time_left > 5:
                 refined, r_ref = run_optimization(best_centers, beta=500, max_iter=100, time_budget=time_left)
                 if r_ref > best_radius:
                     best_radius = r_ref
                     best_centers = refined.copy()

    # Final polish if time remains
    if best_centers is not None and time.time() < deadline - 5:
        # Try a very high beta polish
        refined, r_ref = run_optimization(best_centers, beta=1000, max_iter=50, time_budget=deadline - time.time() - 2)
        if r_ref > best_radius:
            best_radius = r_ref
            best_centers = refined.copy()

    if best_centers is None:
        best_centers = inits[0]
        
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    # Final validation
    final_r = compute_radius(best_centers)
    if final_r <= 0:
        best_centers = inits[0]
        best_centers = np.clip(best_centers, -0.5, 0.5)
        
    return best_centers

# EVOLVE-BLOCK-END
