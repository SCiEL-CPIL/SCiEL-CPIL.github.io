"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s safety margin
    
    # 1. Initialization: Hexagonal Lattice
    # Estimate spacing for N circles in unit disk.
    # Area of disk = pi. Area of one circle ~ pi * r^2.
    # N * pi * r^2 <= pi => r <= 1/sqrt(N).
    # Spacing d = 2r.
    # For hex packing, density is pi/sqrt(12) ~ 0.9069.
    # Effective area per circle in hex lattice is d^2 * sqrt(3)/2.
    # N * d^2 * sqrt(3)/2 ~ pi => d ~ sqrt(2*pi / (N*sqrt(3))).
    d = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
    
    # Generate hex lattice points
    k = int(1.0 / d) + 2
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * d
            y = j * d * np.sqrt(3.0) / 2.0
            points.append([x, y])
    points = np.array(points)
    
    # Filter points inside unit circle (with margin)
    r_pts = np.hypot(points[:, 0], points[:, 1])
    mask = r_pts < 1.0 - d/2.0
    points = points[mask]
    
    # If we have too many, select the closest to center or random?
    # Usually hex lattice gives more than N.
    if len(points) > N:
        # Select N points. To maintain symmetry, we can sort by radius and take the first N?
        # Or just take the first N?
        # Sorting by radius ensures we keep the densest central part.
        idx = np.argsort(r_pts[mask])
        points = points[idx[:N]]
    elif len(points) < N:
        # If not enough, expand k and retry? Or just pad?
        # For N=300, k should be large enough.
        # Let's expand k if needed.
        while len(points) < N:
            k += 1
            points = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * d
                    y = j * d * np.sqrt(3.0) / 2.0
                    points.append([x, y])
            points = np.array(points)
            r_pts = np.hypot(points[:, 0], points[:, 1])
            mask = r_pts < 1.0 - d/2.0
            points = points[mask]
            if len(points) > N:
                idx = np.argsort(r_pts[mask])
                points = points[idx[:N]]
                break
    
    # Ensure shape is (N, 2)
    if len(points) != N:
        # Fallback: just take what we have or pad with zeros (should not happen)
        pass

    # 2. Optimization: Gradient Descent with Soft-Min Surrogate
    # Objective: Minimize S(c) = -1/beta * log(sum(exp(-beta * v_k)))
    # where v_k are min(D_ij/2, W_i).
    # We want to maximize the min, so minimize the negative soft-min.
    
    c = points.copy()
    beta = 50.0  # Start with moderate sharpness
    lr = 0.01    # Learning rate
    best_score = -np.inf
    best_c = c.copy()
    
    # Precompute indices for pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    while time.time() < deadline:
        # Compute distances
        diff = c[i_idx] - c[j_idx]
        dists = np.linalg.norm(diff, axis=1)
        # Avoid division by zero
        dists_safe = np.where(dists < 1e-9, 1e-9, dists)
        
        # Wall distances
        r_c = np.linalg.norm(c, axis=1)
        wall_dists = 1.0 - r_c
        
        # Values v
        v_pairs = dists_safe / 2.0
        v_wall = wall_dists
        
        # Combine into one array for soft-min
        # Note: We need gradients for each coordinate.
        # It's easier to compute the gradient vector directly.
        
        # Softmax weights
        # To avoid underflow/overflow, subtract max
        v_all = np.concatenate([v_pairs, v_wall])
        v_max = np.max(v_all)
        exp_v = np.exp(-beta * (v_all - v_max))
        Z = np.sum(exp_v)
        p_all = exp_v / Z
        
        p_pairs = p_all[:len(v_pairs)]
        p_wall = p_all[len(v_pairs):]
        
        # Gradient of v_pairs w.r.t c
        # d(v_ij)/dc_i = 0.5 * (c_i - c_j) / dist
        # d(v_ij)/dc_j = 0.5 * (c_j - c_i) / dist
        grad_c = np.zeros_like(c)
        
        # Accumulate pair gradients
        # For each pair (i, j):
        # grad_i += p_ij * 0.5 * (c_i - c_j) / dist
        # grad_j += p_ij * 0.5 * (c_j - c_i) / dist
        
        # Vectorized accumulation
        # diff = c[i] - c[j]
        # unit_diff = diff / dists_safe
        # weighted_diff = p_pairs[:, None] * 0.5 * unit_diff
        
        unit_diff = diff / dists_safe[:, None]
        weighted_diff = p_pairs[:, None] * 0.5 * unit_diff
        
        # np.add.at for accumulation
        np.add.at(grad_c, i_idx, weighted_diff)
        np.add.at(grad_c, j_idx, -weighted_diff)
        
        # Wall gradients
        # d(w_i)/dc_i = -c_i / |c_i|
        r_c_safe = np.where(r_c < 1e-9, 1e-9, r_c)
        unit_c = c / r_c_safe[:, None]
        wall_grad = -p_wall[:, None] * unit_c
        grad_c += wall_grad
        
        # Update
        # We are minimizing S(c), which is -softmin.
        # Gradient of S is grad_c calculated above?
        # Let's check signs.
        # S = -1/beta log Z.
        # dS/dc = sum p_k dv_k/dc.
        # We want to decrease S, so move in direction -grad_c.
        # But wait, S is the "cost" of the minimum.
        # If we increase the minimum distance, S decreases?
        # If v_k increases, exp(-beta v_k) decreases, Z decreases, log Z decreases, -log Z increases.
        # So S increases as the minimum distance increases?
        # Let's re-evaluate.
        # We want to MAXIMIZE min(v).
        # Soft-min approximates min(v).
        # So we want to MAXIMIZE SoftMin(v).
        # Let M = SoftMin(v) = -1/beta log(sum(exp(-beta v))).
        # We want to maximize M.
        # So we want to minimize -M.
        # dM/dc = sum p_k dv_k/dc.
        # So we should move in direction +grad_c (gradient ascent on M).
        
        c += lr * grad_c
        
        # Anneal beta
        beta *= 1.01
        
        # Calculate actual score
        dists_now = np.linalg.norm(c[i_idx] - c[j_idx], axis=1)
        r_c_now = np.linalg.norm(c, axis=1)
        min_pair = np.min(dists_now) / 2.0
        min_wall = np.min(1.0 - r_c_now)
        current_score = min(min_pair, min_wall)
        
        if current_score > best_score:
            best_score = current_score
            best_c = c.copy()
            
        # Adaptive learning rate?
        # If score drops significantly, reduce lr?
        # For now, fixed lr with annealing beta should work.
        
    # Final check: ensure validity
    # If any point is outside or overlapping, the score is low.
    # We return the best found.
    
    return best_c
# EVOLVE-BLOCK-END
