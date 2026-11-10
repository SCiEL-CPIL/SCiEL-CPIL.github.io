import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    # Grid definition
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    n_points = X_flat.size

    P_start = np.array([0.0, 0.0, 0.0])
    P_end = np.array([-2.0, 0.0, 90.0])

    def densify_path(path):
        if len(path) == 0:
            return np.array([P_start, P_end])
        
        full_path = np.vstack([P_start, path, P_end])
        
        dense_points = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            delta_a = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(delta_a)))
            for k in range(1, m + 1):
                t = k / m
                p_int = p1 + t * (p2 - p1)
                dense_points.append(p_int)
        
        return np.array(dense_points)

    def score_path_batched(path, batch_size=50):
        dense = densify_path(path)
        M = dense.shape[0]
        if M == 0:
            return 0.0
        
        # Process in batches to save memory
        # A point is in the sofa if it is inside ALL placements
        # We can track which points are still "alive" (inside all placements so far)
        
        alive = np.ones(n_points, dtype=bool)
        
        for start_idx in range(0, M, batch_size):
            end_idx = min(start_idx + batch_size, M)
            batch = dense[start_idx:end_idx]
            B = batch.shape[0]
            
            if not np.any(alive):
                break
            
            # Only check alive points
            X_alive = X_flat[alive]
            Y_alive = Y_flat[alive]
            N_alive = X_alive.size
            
            dx = X_alive[:, None] - batch[None, :, 0]
            dy = Y_alive[:, None] - batch[None, :, 1]
            
            a_rad = np.deg2rad(batch[:, 2])
            cos_a = np.cos(a_rad)[None, :]
            sin_a = np.sin(a_rad)[None, :]
            
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a
            
            # Base corridor:
            # R1: 0 <= u <= 1, -3 <= v <= 1
            # R2: -3 <= u <= 1, 0 <= v <= 1
            
            in_r1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_r2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            
            inside = in_r1 | in_r2
            
            # Point must be inside ALL placements in this batch
            all_inside_batch = np.all(inside, axis=1)
            
            # Update alive mask
            alive_indices = np.where(alive)[0]
            alive[alive_indices[~all_inside_batch]] = False
        
        count = np.count_nonzero(alive)
        return count * 0.0001

    def score_path_fast(path):
        return score_path_batched(path, batch_size=100)

    rng = np.random.default_rng(42)
    
    best_path = None
    best_score = 0.0
    
    # Try different numbers of waypoints
    waypoint_counts = [4, 6, 8, 10]
    
    # Pre-generate initial guesses for each waypoint count
    all_inits = []
    for n_wp in waypoint_counts:
        for restart in range(3):
            # Angles: linearly spaced with noise
            angles = np.linspace(0, 90, n_wp + 2)[1:-1] + rng.normal(0, 3, n_wp)
            # Ensure monotonic
            angles = np.maximum.accumulate(angles)
            angles = np.clip(angles, 1, 89)
            
            # X: Linear from 0 to -2 with noise
            xs_init = np.linspace(0, -2, n_wp + 2)[1:-1] + rng.normal(0, 0.3, n_wp)
            # Y: Around 0, maybe slightly up
            ys_init = rng.normal(0.3, 0.3, n_wp)
            
            path = np.column_stack([xs_init, ys_init, angles])
            all_inits.append(path)
            
            # Evaluate initial
            s = score_path_fast(path)
            if s > best_score:
                best_score = s
                best_path = path.copy()

    # Optimization: Hill climbing with adaptive step size and coordinate ascent
    time_limit_for_search = deadline - 15
    
    for init_path in all_inits:
        if time.time() > time_limit_for_search:
            break
        
        current_path = init_path.copy()
        current_score = score_path_fast(current_path)
        
        if current_score > best_score:
            best_score = current_score
            best_path = current_path.copy()
        
        # Hill climbing with decreasing step size
        sigma = 0.3
        no_improve_count = 0
        max_no_improve = 200
        
        while time.time() < time_limit_for_search:
            if no_improve_count >= max_no_improve:
                break
            
            # Try coordinate ascent first (change one coordinate at a time)
            improved = False
            n_coords = current_path.shape[0] * 3
            
            for _ in range(5):  # Try a few random coordinates
                if time.time() > time_limit_for_search:
                    break
                
                coord_idx = rng.integers(0, n_coords)
                row = coord_idx // 3
                col = coord_idx % 3
                
                # Try perturbing this coordinate
                for delta in [sigma, -sigma, sigma*0.5, -sigma*0.5]:
                    new_path = current_path.copy()
                    new_path[row, col] += delta
                    
                    # For angle coordinates, ensure monotonicity
                    if col == 2:
                        if row > 0:
                            new_path[row, 2] = max(new_path[row, 2], new_path[row-1, 2])
                        if row < current_path.shape[0] - 1:
                            new_path[row, 2] = min(new_path[row, 2], new_path[row+1, 2])
                    
                    new_score = score_path_fast(new_path)
                    
                    if new_score > current_score:
                        current_path = new_path
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_path = current_path.copy()
                        break
            
            if improved:
                no_improve_count = 0
                sigma *= 0.95
            else:
                no_improve_count += 1
                sigma *= 0.98
    
    # Final polish: fine-tune the best path found
    if best_path is not None and time.time() < deadline - 5:
        current_path = best_path.copy()
        current_score = score_path_fast(current_path)
        
        sigma = 0.1
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            
            improved = False
            n_coords = current_path.shape[0] * 3
            
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                
                coord_idx = rng.integers(0, n_coords)
                row = coord_idx // 3
                col = coord_idx % 3
                
                for delta in [sigma, -sigma]:
                    new_path = current_path.copy()
                    new_path[row, col] += delta
                    
                    if col == 2:
                        if row > 0:
                            new_path[row, 2] = max(new_path[row, 2], new_path[row-1, 2])
                        if row < current_path.shape[0] - 1:
                            new_path[row, 2] = min(new_path[row, 2], new_path[row+1, 2])
                    
                    new_score = score_path_fast(new_path)
                    
                    if new_score > current_score:
                        current_path = new_path
                        current_score = new_score
                        improved = True
                        break
            
            if improved:
                sigma *= 0.9
            else:
                sigma *= 0.95
        
        if current_score > best_score:
            best_score = current_score
            best_path = current_path.copy()
    
    return best_path
