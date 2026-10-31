def optimize_local(points, time_limit):
    start_time = time.time()
    points = points.copy()
    best_points = points.copy()
    best_area = min_triangle_area(best_points)[0]
    
    step_size = 0.05
    prev_area = best_area
    
    for _ in range(10000):
        if time.time() - start_time > time_limit:
            break
            
        current_area, active_indices = min_triangle_area(points)
        if current_area > best_area:
            best_area = current_area
            best_points = points.copy()
            prev_area = current_area
            # Increase step size slightly
            step_size = min(step_size * 1.05, 0.2)
        else:
            # Decrease step size
            step_size *= 0.9
            if step_size < 1e-4:
                break

        # Compute gradients as before
        grads = np.zeros_like(points)
        counts = np.zeros(N)
        
        for (i, j, k) in active_indices:
            p_i, p_j, p_k = points[i], points[j], points[k]
            det = np.cross(p_j - p_i, p_k - p_i)
            sign = 1 if det >= 0 else -1
            
            grad_i = 0.5 * sign * np.array([p_j[1] - p_k[1], p_k[0] - p_j[0]])
            grads[i] += grad_i
            counts[i] += 1
            
            grad_j = 0.5 * sign * np.array([p_k[1] - p_i[1], p_i[0] - p_k[0]])
            grads[j] += grad_j
            counts[j] += 1
            
            grad_k = 0.5 * sign * np.array([-(p_j[1] - p_i[1]), p_j[0] - p_i[0]])
            grads[k] += grad_k
            counts[k] += 1
            
        mask = counts > 0
        grads[mask] /= counts[mask, np.newaxis]
        
        # Store old points
        old_points = points.copy()
        
        # Update points
        for idx in np.where(mask)[0]:
            new_p = points[idx] + step_size * grads[idx]
            if is_inside_triangle(new_p, V0, V1, V2):
                points[idx] = new_p
            else:
                proj_p = project_to_triangle(new_p, V0, V1, V2)
                points[idx] = proj_p
                
        # Check improvement
        new_area, _ = min_triangle_area(points)
        if new_area < prev_area * 0.99: # Significant drop
            points = old_points
            step_size *= 0.5
            
    return best_points, best_area
