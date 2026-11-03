<<<<<<< SEARCH
def polish(points, time_budget):
    """
    Gradient-based polish to refine the packing.
    """
    start_time = time.time()
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)
    
    best_points = points.copy()
    best_score = compute_score(points)
    
    # Adaptive k and lr
    k = 100.0
    lr = 0.005
    
    # We will run until time_budget is exhausted
    while time.time() - start_time < time_budget:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_matrix = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists_matrix, np.inf)
        
        pair_dists = dists_matrix[i_idx, j_idx]
        rad = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - rad
        
        all_dists = np.concatenate([pair_dists, bound_dists])
        current_score = np.min(all_dists)
        
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
        
        # Soft min gradient
        shifted_dists = all_dists - current_score
        # Only consider constraints close to the minimum to focus effort
        mask_close = shifted_dists < 0.05
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
        if grad_norm > 1e-9:
            grads /= grad_norm
            points += lr * grads
            
        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]
            
        # Decay lr and increase k slightly
        lr *= 0.995
        k *= 1.01
        
    return best_points, best_score
=======
def polish(points, time_budget):
    """
    High-precision polish with tight tolerances.
    """
    start_time = time.time()
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)
    
    best_points = points.copy()
    best_score = compute_score(points)
    
    k = 200.0  # Higher k for sharper min
    lr = 0.001  # Smaller lr for precision
    
    while time.time() - start_time < time_budget:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_matrix = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists_matrix, np.inf)
        
        pair_dists = dists_matrix[i_idx, j_idx]
        rad = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - rad
        
        all_dists = np.concatenate([pair_dists, bound_dists])
        current_score = np.min(all_dists)
        
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
        
    return best_points, best_score
>>>>>>> REPLACE

