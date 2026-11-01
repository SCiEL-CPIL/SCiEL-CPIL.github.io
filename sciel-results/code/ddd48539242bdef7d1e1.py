import numpy as np
from scipy.optimize import minimize

N = 200

def construct():
    # Deadline guard
    import time
    start_time = time.time()
    deadline = 145.0 # 5s safety margin
    
    # 1. Initialization
    # Use a hexagonal grid for better initial packing
    # 14 rows x 15 cols = 210 points
    rows, cols = 14, 15
    # Scale to fit in unit square [-0.5, 0.5]
    # Height = rows * dy, Width = cols * dx
    # dx = a, dy = a * sqrt(3)/2
    # We want to fit 200 points.
    # Let's generate a grid and clip
    a = 1.0 / (cols * 1.1) # Rough estimate
    pts = []
    for r in range(rows):
        for c in range(cols):
            x = (c + (0.5 if r % 2 == 1 else 0)) * a
            y = r * a * np.sqrt(3) / 2
            pts.append([x, y])
    pts = np.array(pts)
    
    # Center the grid
    pts[:, 0] -= np.mean(pts[:, 0])
    pts[:, 1] -= np.mean(pts[:, 1])
    
    # Clip to square and remove duplicates/outliers, take first 200
    # Actually, just take a simple grid if hex is too complex to init perfectly
    # Let's stick to a simple grid with random jitter for robustness
    m = int(np.ceil(np.sqrt(N))) # 15
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # Take 200 from 15x15=225
    # Randomly shuffle to break symmetry?
    np.random.seed(42)
    idx = np.random.permutation(len(grid))[:N]
    x0 = grid[idx].copy()
    
    # Add small random jitter
    x0 += np.random.normal(0, 0.01, x0.shape)
    
    # Clip to bounds
    x0[:, 0] = np.clip(x0[:, 0], -0.49, 0.49)
    x0[:, 1] = np.clip(x0[:, 1], -0.49, 0.49)
    
    x = x0.flatten()
    
    def objective(flat_x):
        x = flat_x.reshape(-1, 2)
        # Compute all pairwise distances
        # Efficiently:
        # diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=-1)
        # This is O(N^2) memory, 200x200 is fine.
        
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        
        # Wall distances
        # dist to left: x[:, 0] + 0.5
        # dist to right: 0.5 - x[:, 0]
        # dist to bottom: x[:, 1] + 0.5
        # dist to top: 0.5 - x[:, 1]
        d_left = x[:, 0] + 0.5
        d_right = 0.5 - x[:, 0]
        d_bot = x[:, 1] + 0.5
        d_top = 0.5 - x[:, 1]
        
        # We want min of all dists (i<j) and wall dists
        # Set diagonal to inf to ignore self-distance
        np.fill_diagonal(dists, np.inf)
        
        # Get upper triangle
        triu_idx = np.triu_indices(N, k=1)
        pair_dists = dists[triu_idx]
        
        wall_dists = np.concatenate([d_left, d_right, d_bot, d_top])
        
        all_dists = np.concatenate([pair_dists, wall_dists])
        
        # Soft-min
        # m = -tau * log(sum(exp(-d/tau)))
        # We want to maximize m, so minimize -m
        tau = current_tau[0]
        # Numerical stability: subtract min
        d_min = np.min(all_dists)
        exp_terms = np.exp(-(all_dists - d_min) / tau)
        soft_min_val = d_min - tau * np.log(np.sum(exp_terms))
        
        # Return negative for minimization
        return -soft_min_val
    
    def objective_grad(flat_x):
        # Gradient of -softmin
        # softmin(x) = x_min - tau * log(sum exp(-x_i/tau + x_min/tau))
        # d/dx_i softmin = -1/tau * exp(-x_i/tau) / sum exp(-x_j/tau)
        # So d/dx_i (-softmin) = 1/tau * exp(-x_i/tau) / sum exp(-x_j/tau)
        
        x = flat_x.reshape(-1, 2)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        d_left = x[:, 0] + 0.5
        d_right = 0.5 - x[:, 0]
        d_bot = x[:, 1] + 0.5
        d_top = 0.5 - x[:, 1]
        
        triu_idx = np.triu_indices(N, k=1)
        pair_dists = dists[triu_idx]
        
        wall_dists = np.concatenate([d_left, d_right, d_bot, d_top])
        
        all_dists = np.concatenate([pair_dists, wall_dists])
        
        tau = current_tau[0]
        d_min = np.min(all_dists)
        exp_terms = np.exp(-(all_dists - d_min) / tau)
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp / tau
        
        grad = np.zeros_like(flat_x)
        
        # We need to map weights back to gradient of distance wrt positions
        # This is complex to do directly with the flattened array without careful indexing.
        # For simplicity and speed within time limit, let's rely on numerical gradients
        # or use a simpler approach: just minimize using L-BFGS-B which can do finite diffs
        # but N=200, 400 vars. Finite diffs might be slow.
        # Let's implement the analytical gradient properly or use a simpler force model.
        
        # Alternative: Use a "force" based simulation (Molecular Dynamics) which is very fast.
        # Force = sum of repulsive forces from neighbors and walls.
        # This is equivalent to gradient ascent of sum of 1/d or exp(-d).
        
        pass # We will switch strategy to MD if analytic grad is too hard to code quickly.

    # Strategy Change: Molecular Dynamics (Repulsive Forces)
    # It's more robust and faster for this type of problem.
    # Minimize potential energy: U = sum_{i<j} 1/d_{ij}^2 + sum_i 1/d_{wall}^2
    # Or just use L-BFGS on the soft-min but let it compute gradients numerically?
    # L-BFGS-B with finite differences for 400 variables might be too slow if we do many iterations.
    
    # Let's use the Soft-Min with Analytical Gradient.
    # Gradient of d_{ij} wrt x_i: (x_i - x_j) / d_{ij}
    # Gradient of d_{ij} wrt x_j: (x_j - x_i) / d_{ij}
    
    # We will define the gradient function properly.
    
    current_tau = [0.1]
    
    def soft_min_obj_grad(flat_x):
        x = flat_x.reshape(-1, 2)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1) + 1e-12)
        np.fill_diagonal(dists, np.inf)
        
        d_left = x[:, 0] + 0.5
        d_right = 0.5 - x[:, 0]
        d_bot = x[:, 1] + 0.5
        d_top = 0.5 - x[:, 1]
        
        triu_i, triu_j = np.triu_indices(N, k=1)
        pair_dists = dists[triu_i, triu_j]
        
        # Collect all distances
        all_dists = np.concatenate([pair_dists, d_left, d_right, d_bot, d_top])
        
        tau = current_tau[0]
        d_min = np.min(all_dists)
        exp_terms = np.exp(-(all_dists - d_min) / tau)
        sum_exp = np.sum(exp_terms)
        w = exp_terms / sum_exp / tau
        
        # Gradient of objective (which is -softmin)
        # d(-softmin)/dx = sum_k w_k * (d dist_k / dx)
        
        grad = np.zeros_like(flat_x)
        
        # Pair contributions
        # For each pair (i, j), weight w_k
        # d dist_ij / d x_i = (x_i - x_j) / dist_ij
        # d dist_ij / d x_j = (x_j - x_i) / dist_ij
        # Note: w is positive. Objective is -softmin.
        # softmin = min approx. d(softmin)/dx = sum w_k d(dist_k)/dx.
        # So d(-softmin)/dx = - sum w_k d(dist_k)/dx.
        
        # Vectorized pair update
        # We have weights w[0 : N_pairs] corresponding to triu pairs.
        # We need to accumulate gradients for i and j.
        
        # Create an array for pair gradients
        # delta_ij = x_i - x_j
        deltas = x[triu_i] - x[triu_j]
        # unit vectors
        unit_deltas = deltas / pair_dists[:, np.newaxis]
        # weights scaled
        w_pairs = w[:len(pair_dists)]
        
        # grad_i += w_pairs * unit_deltas
        # grad_j -= w_pairs * unit_deltas
        
        # Use bincount/add for efficient accumulation
        # Since N is small (200), we can just loop or use sparse logic.
        # Looping 200*200/2 = 20000 iterations is fast in Python if vectorized inside? No, loop is slow.
        # Let's use np.add.at
        
        np.add.at(grad, triu_i, w_pairs[:, np.newaxis] * unit_deltas)
        np.add.at(grad, triu_j, -w_pairs[:, np.newaxis] * unit_deltas)
        
        # Wall contributions
        # Indices in all_dists:
        # pairs: 0 .. N_pairs-1
        # left: N_pairs .. N_pairs+N-1
        # right: N_pairs+N .. N_pairs+2N-1
        # bot: N_pairs+2N .. N_pairs+3N-1
        # top: N_pairs+3N .. N_pairs+4N-1
        
        N_pairs = len(pair_dists)
        w_left = w[N_pairs : N_pairs+N]
        w_right = w[N_pairs+N : N_pairs+2*N]
        w_bot = w[N_pairs+2*N : N_pairs+3*N]
        w_top = w[N_pairs+3*N : N_pairs+4*N]
        
        # d(d_left)/d(x,0) = 1
        # d(d_right)/d(x,0) = -1
        # d(d_bot)/d(x,1) = 1
        # d(d_top)/d(x,1) = -1
        
        grad[:, 0] += w_left
        grad[:, 0] -= w_right
        grad[:, 1] += w_bot
        grad[:, 1] -= w_top
        
        # The objective we are minimizing is -softmin.
        # We calculated d(softmin)/dx = sum w d(dist)/dx.
        # So d(-softmin)/dx = - sum w d(dist)/dx.
        # Our `grad` currently holds sum w d(dist)/dx.
        # So we need to negate it.
        return -grad
        
    # Optimization Loop
    temps = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    
    for t in temps:
        if time.time() > deadline:
            break
        current_tau[0] = t
        # Use L-BFGS-B
        # Bounds
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        res = minimize(
            soft_min_obj_grad, 
            x, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 1000}
        )
        x = res.x
        
    # Final Polish
    # Very small tau
    current_tau[0] = 0.0005
    bounds = [(-0.5, 0.5)] * (2 * N)
    if time.time() < deadline - 5:
        res = minimize(
            soft_min_obj_grad, 
            x, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        x = res.x

    return x.reshape(-1, 2)
