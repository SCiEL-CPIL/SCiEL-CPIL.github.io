import numpy as np
import time
import math

N = 300

def compute_score(points):
    """
    Compute the packing radius for a set of points.
    Optimized to be faster by using squared distances where possible, though sqrt is needed for final score.
    """
    rad = np.linalg.norm(points, axis=1)
    min_bound = 1.0 - np.max(rad)
    
    # Efficient distance calculation
    # Use broadcasting
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # We need the min distance, which is sqrt(min dist_sq)
    min_pair = np.sqrt(np.min(dists_sq))
    
    score = min(min_pair, min_bound)
    return score

def soft_min_gradient(points, k=50.0):
    """
    Compute gradient of soft-min of all distances (pairwise and boundary).
    Returns (gradient, soft_min_value)
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_matrix = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists_matrix, np.inf)
    
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    pair_dists = dists_matrix[i_idx, j_idx]
    
    rad = np.linalg.norm(points, axis=1)
    bound_dists = 1.0 - rad
    
    all_dists = np.concatenate([pair_dists, bound_dists])
    n_total = len(all_dists)
    
    # Soft min using log-sum-exp
    min_val = np.min(all_dists)
    shifted = all_dists - min_val
    exp_vals = np.exp(-k * shifted)
    sum_exp = np.sum(exp_vals)
    
    if sum_exp == 0:
        return np.zeros_like(points), min_val
    
    weights = exp_vals / sum_exp
    soft_min_val = min_val - np.log(sum_exp) / k
    
    grads = np.zeros_like(points)
    
    # Pairwise gradient
    pair_weights = weights[:len(pair_dists)]
    mask_nonzero = pair_dists > 1e-9
    if np.any(mask_nonzero):
        v_ij = diff[i_idx[mask_nonzero], j_idx[mask_nonzero]] / pair_dists[mask_nonzero, np.newaxis]
        w_ij = pair_weights[mask_nonzero, np.newaxis]
        forces = np.zeros((N, 2))
        np.add.at(forces, i_idx[mask_nonzero], v_ij * w_ij)
        np.add.at(forces, j_idx[mask_nonzero], -v_ij * w_ij)
        grads += forces
    
    # Boundary gradient
    bound_weights = weights[len(pair_dists):]
    rad_safe = np.where(rad > 1e-9, rad, 1.0)
    bound_grads = -(points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
    grads += bound_grads
    
    return grads, soft_min_val

def gradient_optimize(points, time_budget):
    """
    Gradient-based optimization using soft-min surrogate.
    Optimized to reduce overhead by checking exact score less frequently.
    """
    start_time = time.time()
    
    best_points = points.copy()
    best_score = compute_score(points)
    best_soft_val = -np.inf
    
    k = 30.0  # Soft-min parameter
    lr = 0.002
    iter_count = 0
    
    while time.time() - start_time < time_budget:
        # Check time frequently
        if time.time() - start_time > time_budget - 1.0:
            break
            
        grads, soft_val = soft_min_gradient(points, k)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1e-12:
            grads = grads / grad_norm
            points += lr * grads
        
        # Project inside container
        rad_new = np.linalg.norm(points, axis=1)
        mask_out = rad_new > 1.0
        if np.any(mask_out):
            points[mask_out] /= rad_new[mask_out, np.newaxis]
        
        # Track progress using soft_val (faster)
        if soft_val > best_soft_val:
            best_soft_val = soft_val
            # Only compute exact score if soft_val has improved significantly or periodically
            if iter_count % 5 == 0 or soft_val > best_score + 1e-4:
                current_score = compute_score(points)
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
        
        # Adapt learning rate and k
        lr *= 0.998
        k *= 1.005
        iter_count += 1
        
    # Final exact score check
    final_score = compute_score(points)
    if final_score > best_score:
        best_score = final_score
        best_points = points.copy()
        
    return best_points, best_score

def polish(points, time_budget):
    """
    High-precision polish with tight tolerances.
    Optimized to reduce overhead.
    """
    start_time = time.time()
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)
    
    best_points = points.copy()
    best_score = compute_score(points)
    best_soft_val = -np.inf
    
    k = 200.0  # Higher k for sharper min
    lr = 0.001  # Smaller lr for precision
    iter_count = 0
    
    while time.time() - start_time < time_budget:
        if time.time() - start_time > time_budget - 1.0:
            break
        
        # Compute distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists_matrix = np.sqrt(dists_sq)
        
        pair_dists = dists_matrix[i_idx, j_idx]
        rad = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - rad
        
        all_dists = np.concatenate([pair_dists, bound_dists])
        current_score = np.min(all_dists)
        
        # Track progress
        if current_score > best_soft_val:
            best_soft_val = current_score
            if iter_count % 5 == 0 or current_score > best_score + 1e-5:
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
        
        # Focus on constraints very close to minimum
        shifted_dists = all_dists - current_score
        mask_close = shifted_dists < 0.01  # Tighter threshold
        if not np.any(mask_close):
            break
            
        close_dists = shifted_dists[mask_close]
        exp_weights = np.exp(-k * close_dists)
        sum_weights = np.sum(exp_weights)
        
        if sum_weights == 0:
            break
            
        weights_full = np.zeros_like(all_dists)
        weights_full[mask_close] = exp_weights / sum_weights
        
        grads = np.zeros_like(points)
        
        pair_weights = weights_full[:num_pairs]
        d_pair = dists_matrix[i_idx, j_idx]
        mask_nonzero = d_pair > 1e-9
        if np.any(mask_nonzero):
            v_ij = diff[i_idx[mask_nonzero], j_idx[mask_nonzero]] / d_pair[mask_nonzero, np.newaxis]
            w_ij = pair_weights[mask_nonzero, np.newaxis]
            forces = np.zeros((N, 2))
            np.add.at(forces, i_idx[mask_nonzero], v_ij * w_ij)
            np.add.at(forces, j_idx[mask_nonzero], -v_ij * w_ij)
            grads += forces
            
        bound_weights = weights_full[num_pairs:]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = -(points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1e-12:
            grads /= grad_norm
            points += lr * grads
            
        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]
            
        # Very slow decay for precision
        lr *= 0.999
        k *= 1.002
        iter_count += 1
        
    # Final exact score check
    final_score = compute_score(points)
    if final_score > best_score:
        best_score = final_score
        best_points = points.copy()
        
    return best_points, best_score

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s kill
    
    # 1. Initialization with better hexagonal packing
    # Use density-based spacing
    area_per_circle = np.pi / N
    s_init = np.sqrt(2 * area_per_circle / np.sqrt(3))
    
    # Try multiple spacings and pick best
    best_points = None
    best_score = -np.inf
    
    for scale in [0.95, 1.0, 1.05]:
        s_gen = s_init * scale
        y_step = s_gen * np.sqrt(3) / 2
        max_j = int(1.0 / y_step) + 2
        max_i = int(1.0 / s_gen) + 2
        
        points_list = []
        for j in range(-max_j, max_j + 1):
            y = j * y_step
            offset = 0 if j % 2 == 0 else s_gen / 2
            for i in range(-max_i, max_i + 1):
                x = i * s_gen + offset
                if np.hypot(x, y) < 0.99:
                    points_list.append([x, y])
                    
        if len(points_list) < N:
            continue
            
        points = np.array(points_list)
        
        if len(points) > N:
            # Keep points closest to center for better packing
            dists_to_center = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists_to_center)[:N]
            points = points[idx]
            
        score = compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    if best_points is None:
        # Fallback: random initialization
        best_points = np.random.uniform(-0.9, 0.9, (N, 2))
        best_score = compute_score(best_points)
    
    # 2. Gradient Optimization Phase
    opt_time = time_limit * 0.6
    opt_points, opt_score = gradient_optimize(best_points, opt_time)
    
    if opt_score > best_score:
        best_score = opt_score
        best_points = opt_points.copy()
    
    # 3. Polish Phase
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 2.0:
        final_points, final_score = polish(best_points, remaining_time)
        if final_score > best_score:
            best_score = final_score
            best_points = final_points
            
    return best_points
