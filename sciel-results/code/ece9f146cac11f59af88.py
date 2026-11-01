    # Initial diff/dist
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dist_safe = np.where(dist > 1e-6, dist, 1e-6)
    
    while time.time() - start_time < time_limit:
        # Calculate forces using current diff/dist
        overlap = np.maximum(0, 2 * r - dist)
        dir_vec = diff / dist_safe[:, :, np.newaxis]
        force_rep = dir_vec * overlap[:, :, np.newaxis]
        F_rep = np.sum(force_rep, axis=1)
        
        r_norm = np.linalg.norm(points, axis=1)
        boundary_overlap = np.maximum(0, r_norm - (1 - r))
        dir_bound = points / np.where(r_norm > 1e-6, r_norm, 1e-6)[:, np.newaxis]
        F_bound = dir_bound * boundary_overlap[:, np.newaxis]
        
        F = F_rep + F_bound
        
        points += eta * F
        
        # Project
        r_norm_new = np.linalg.norm(points, axis=1)
        out_mask = r_norm_new > 1.0
        if np.any(out_mask):
            points[out_mask] /= r_norm_new[out_mask, np.newaxis]
            
        # Update diff/dist for next iteration and r calculation
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        dist_safe = np.where(dist > 1e-6, dist, 1e-6)
        
        # Calculate r
        min_dist = np.min(dist[mask])
        r_boundary = 1 - np.max(r_norm_new) # Use r_norm_new? 
        # r_norm_new was calculated before projection? No, after update, before projection.
        # We need r_norm after projection.
        r_norm_final = np.linalg.norm(points, axis=1)
        r_boundary = 1 - np.max(r_norm_final)
        
        r_new = min(min_dist / 2, r_boundary)
        
        if r_new > r:
            r = r_new
            if r > best_r:
                best_r = r
                best_points = points.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                points += np.random.normal(0, 0.01, points.shape)
                no_improve_count = 0
                eta *= 0.9
                if eta < 0.0001:
                    eta = 0.01
                    points += np.random.normal(0, 0.02, points.shape)
