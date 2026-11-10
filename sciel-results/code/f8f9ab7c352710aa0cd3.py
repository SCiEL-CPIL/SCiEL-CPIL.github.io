import numpy as np
import time

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin (630s kill, leave 40s buffer)

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

    def score_path_fast(path):
        dense = densify_path(path)
        M = dense.shape[0]
        if M == 0:
            return 0.0
        
        alive = np.ones(n_points, dtype=bool)
        
        # Precompute cos/sin for all dense points
        a_rad = np.deg2rad(dense[:, 2])
        cos_all = np.cos(a_rad)
        sin_all = np.sin(a_rad)
        
        for i in range(M):
            if not np.any(alive):
                break
            
            x0, y0, a = dense[i]
            c, s = cos_all[i], sin_all[i]
            
            X_alive = X_flat[alive]
            Y_alive = Y_flat[alive]
            
            dx = X_alive - x0
            dy = Y_alive - y0
            
            u = dx * c + dy * s
            v = -dx * s + dy * c
            
            # Base corridor:
            # R1: 0 <= u <= 1, -3 <= v <= 1
            # R2: -3 <= u <= 1, 0 <= v <= 1
            
            in_r1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_r2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            
            inside = in_r1 | in_r2
            
            # Update alive mask
            alive_indices = np.where(alive)[0]
            alive[alive_indices[~inside]] = False
        
        count = np.count_nonzero(alive)
        return count * 0.0001

    rng = np.random.default_rng(42)
    
    def get_seeded_normal(mean, std, size, seed_val):
        """Helper to get normal random numbers with a specific seed."""
        return np.random.default_rng(seed_val).normal(mean, std, size)

    best_path = None
    best_score = 0.0
    
    # Better initial guesses: use a known good path structure
    # The optimal path for moving a sofa around a corner typically follows
    # a curve where the inner corner traces a specific trajectory
    
    def make_initial_path(n_wp, seed_offset=0):
        """Create a reasonable initial path."""
        # Angles from 0 to 90
        angles = np.linspace(0, 90, n_wp + 2)[1:-1]
        angles += get_seeded_normal(0, 2, n_wp, seed_offset)
        angles = np.maximum.accumulate(angles)
        angles = np.clip(angles, 1, 89)
        
        # Position: the inner corner should move from (0,0) to (-2,0)
        # while rotating. A good approximation is a circular arc.
        # At angle a, the corner position can be approximated.
        
        # For a 90-degree turn, the sofa moves along a path where
        # x goes from 0 to -2 and y stays near 0, with some curvature
        
        t = np.linspace(0, 1, n_wp)
        # Use a smooth curve
        xs_init = -2.0 * t + get_seeded_normal(0, 0.2, n_wp, seed_offset+100)
        # Y position: slight arc
        ys_init = 0.5 * np.sin(np.pi * t) + get_seeded_normal(0, 0.2, n_wp, seed_offset+200)
        
        return np.column_stack([xs_init, ys_init, angles])
    
    # Try several initial paths with different structures
    init_paths = []
    
    # Type 1: Nearly linear path
    for n_wp in [5, 7, 9]:
        for restart in range(4):
            path = make_initial_path(n_wp, seed_offset=restart*10)
            init_paths.append(path)
            s = score_path_fast(path)
            if s > best_score:
                best_score = s
                best_path = path.copy()
    
    # Type 2: More curved paths
    for n_wp in [6, 8]:
        for restart in range(3):
            angles = np.linspace(0, 90, n_wp + 2)[1:-1]
            angles += get_seeded_normal(0, 3, n_wp, restart*20+50)
            angles = np.maximum.accumulate(angles)
            angles = np.clip(angles, 1, 89)
            
            t = np.linspace(0, 1, n_wp)
            # Different curve shapes
            curve_type = restart % 3
            if curve_type == 0:
                xs_init = -2.0 * t**1.5
                ys_init = 0.8 * np.sin(np.pi * t**0.7)
            elif curve_type == 1:
                xs_init = -2.0 * t**0.8
                ys_init = 0.6 * np.sin(np.pi * t**1.2)
            else:
                xs_init = -2.0 * np.sin(np.pi * t / 2)
                ys_init = 0.7 * t * (1 - t) * 4
            
            xs_init += get_seeded_normal(0, 0.15, n_wp, restart*30+60)
            ys_init += get_seeded_normal(0, 0.15, n_wp, restart*40+70)
            
            path = np.column_stack([xs_init, ys_init, angles])
            init_paths.append(path)
            s = score_path_fast(path)
            if s > best_score:
                best_score = s
                best_path = path.copy()
    
    # Hill climbing optimization
    time_limit_for_search = deadline - 20
    
    for init_path in init_paths:
        if time.time() > time_limit_for_search:
            break
        
        current_path = init_path.copy()
        current_score = score_path_fast(current_path)
        
        if current_score > best_score:
            best_score = current_score
            best_path = current_path.copy()
        
        # Adaptive hill climbing
        sigma = 0.25
        no_improve_count = 0
        max_no_improve = 150
        n_coords = current_path.shape[0] * 3
        
        while time.time() < time_limit_for_search:
            if no_improve_count >= max_no_improve:
                break
            
            improved = False
            
            # Try multiple random coordinate perturbations
            n_trials = 8
            for _ in range(n_trials):
                if time.time() > time_limit_for_search:
                    break
                
                coord_idx = rng.integers(0, n_coords)
                row = coord_idx // 3
                col = coord_idx % 3
                
                # Try several step sizes
                for delta in [sigma, -sigma, sigma*0.6, -sigma*0.6, sigma*0.3, -sigma*0.3]:
                    new_path = current_path.copy()
                    new_path[row, col] += delta
                    
                    # Enforce angle monotonicity
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
                sigma *= 0.92
            else:
                no_improve_count += 1
                sigma *= 0.97
    
    # Final fine-tuning
    if best_path is not None and time.time() < deadline - 10:
        current_path = best_path.copy()
        current_score = score_path_fast(current_path)
        
        sigma = 0.08
        for _ in range(300):
            if time.time() > deadline - 10:
                break
            
            improved = False
            n_coords = current_path.shape[0] * 3
            
            for _ in range(12):
                if time.time() > deadline - 10:
                    break
                
                coord_idx = rng.integers(0, n_coords)
                row = coord_idx // 3
                col = coord_idx % 3
                
                for delta in [sigma, -sigma, sigma*0.5, -sigma*0.5]:
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
                sigma *= 0.88
            else:
                sigma *= 0.94
        
        if current_score > best_score:
            best_score = current_score
            best_path = current_path.copy()
    
    if best_path is None:
        # Fallback: simple linear path
        best_path = np.array([[-0.5, 0.3, 22.5], [-1.0, 0.5, 45.0], [-1.5, 0.3, 67.5]])
    
    return best_path
