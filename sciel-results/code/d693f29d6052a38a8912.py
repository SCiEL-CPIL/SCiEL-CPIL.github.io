import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 150s kill
    
    best_score = 0.0
    best_config = None
    
    # Precompute indices for vectorized overlap calculation
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def validity_check(x, y, r):
        """Returns True if configuration is valid, False otherwise"""
        n = len(r)
        if np.any(r < -1e-12):
            return False
        if np.any(x - r < -1e-9) or np.any(y - r < -1e-9):
            return False
        if np.any(x + r > 1 + 1e-9) or np.any(y + r > 1 + 1e-9):
            return False
        # Vectorized overlap check
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        if np.any(dist < r[i_idx] + r[j_idx] - 1e-9):
            return False
        return True
    
    def make_valid(x, y, r):
        """Ensure configuration is valid by shrinking radii if needed"""
        r = r.copy()
        # First ensure bounds
        for i in range(N):
            max_r = min(x[i], 1 - x[i], y[i], 1 - y[i])
            if r[i] > max_r:
                r[i] = max(0, max_r)
        
        # Iteratively fix overlaps
        for _ in range(200):
            # Vectorized overlap calculation
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            dist = np.sqrt(dx*dx + dy*dy)
            overlaps = r[i_idx] + r[j_idx] - dist
            
            max_overlap = np.max(overlaps)
            if max_overlap <= 0:
                break
            
            # Find the pair with max overlap
            max_pair_idx = np.argmax(overlaps)
            max_i = i_idx[max_pair_idx]
            max_j = j_idx[max_pair_idx]
            
            # Shrink the larger circle by half the overlap plus a small buffer
            shrink = max_overlap / 2 + 1e-7
            if r[max_i] >= r[max_j]:
                r[max_i] = max(0, r[max_i] - shrink)
            else:
                r[max_j] = max(0, r[max_j] - shrink)
        
        # Final safety: if still invalid, scale down all radii
        if not validity_check(x, y, r):
            r *= 0.95
            for _ in range(50):
                if validity_check(x, y, r):
                    break
                r *= 0.9
        
        return x, y, r
    
    def maximize_radii(x, y, r, iterations=20):
        """Try to increase each radius to its maximum possible value"""
        eps = 1e-10
        r = r.copy()
        for _ in range(iterations):
            if time.time() > deadline:
                break
            changed = False
            for i in range(N):
                # Max radius from boundary
                r_bound = min(x[i], 1 - x[i], y[i], 1 - y[i])
                # Max radius from neighbors
                r_neighbor = 0.5
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    r_neighbor = min(r_neighbor, max(0, dist - r[j] - eps))
                
                new_r = min(r_bound, r_neighbor)
                if new_r > r[i] + 1e-12:
                    r[i] = new_r
                    changed = True
            
            if not changed:
                break
        
        # Ensure validity after maximization
        if not validity_check(x, y, r):
            x, y, r = make_valid(x, y, r)
        
        return x, y, r
    
    def force_relaxation(x, y, r, iterations=100):
        """
        Simple force-directed relaxation to help escape local minima.
        Pushes overlapping circles apart and pulls them toward valid positions.
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        for it in range(iterations):
            if time.time() > deadline:
                break
            
            # Calculate forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Repulsion between overlapping circles
            dx = x[i_idx] - x[j_idx]
            dy = y[i_idx] - y[j_idx]
            dist = np.sqrt(dx*dx + dy*dy)
            dist = np.maximum(dist, 1e-10)  # Avoid division by zero
            
            overlap = r[i_idx] + r[j_idx] - dist
            overlap = np.maximum(overlap, 0)
            
            # Force magnitude proportional to overlap
            force_mag = overlap * 10.0
            force_x = force_mag * dx / dist
            force_y = force_mag * dy / dist
            
            # Accumulate forces
            for k in range(len(force_mag)):
                i = i_idx[k]
                j = j_idx[k]
                fx[i] += force_x[k]
                fy[i] += force_y[k]
                fx[j] -= force_x[k]
                fy[j] -= force_y[k]
            
            # Boundary forces
            for i in range(N):
                # Left boundary
                if x[i] < r[i]:
                    fx[i] += (r[i] - x[i]) * 50.0
                # Right boundary
                if x[i] > 1 - r[i]:
                    fx[i] -= (x[i] - (1 - r[i])) * 50.0
                # Bottom boundary
                if y[i] < r[i]:
                    fy[i] += (r[i] - y[i]) * 50.0
                # Top boundary
                if y[i] > 1 - r[i]:
                    fy[i] -= (y[i] - (1 - r[i])) * 50.0
            
            # Apply forces with damping
            damping = 0.5
            x += fx * damping * 0.01
            y += fy * damping * 0.01
            
            # Clip to bounds
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
        
        return x, y, r
    
    def objective(params):
        """
        Objective function for L-BFGS-B.
        We want to maximize sum(r), so we minimize -sum(r).
        We add penalties for boundary violations and circle overlaps.
        Uses smooth soft-min approximation for better gradient behavior.
        """
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Ensure radii are positive
        r_eff = np.maximum(r, 1e-6)
        
        # 1. Boundary Penalties (smooth)
        # For each constraint c_i >= 0, penalty is soft_min(c_i, 0)
        # soft_min(a, 0) = -logsumexp(-a, 0) / beta for beta > 0
        # But simpler: use max(0, -c)^2 which is smooth enough for L-BFGS-B
        # Actually, let's use a smoother approximation:
        # penalty = -beta * log(1 + exp(-c/beta)) for c >= 0
        # For c < 0, this is approximately -c + small
        
        # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
        c1 = x - r_eff
        c2 = 1 - x - r_eff
        c3 = y - r_eff
        c4 = 1 - y - r_eff
        
        # Smooth penalty: -beta * log(1 + exp(-c/beta))
        beta = 0.01
        def smooth_pen(c):
            # For numerical stability
            z = -c / beta
            # log(1 + exp(z)) = max(0, z) + log(1 + exp(-|z|))
            abs_z = np.abs(z)
            log_term = np.maximum(0, z) + np.log1p(np.exp(-abs_z))
            return beta * log_term
        
        boundary_pen = np.sum(smooth_pen(c1) + smooth_pen(c2) + smooth_pen(c3) + smooth_pen(c4))
        
        # 2. Overlap Penalties (Vectorized, smooth)
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        d = np.sqrt(dx*dx + dy*dy)
        # Constraint: d >= r_i + r_j, so c = d - r_i - r_j >= 0
        c_overlap = d - r_eff[i_idx] - r_eff[j_idx]
        overlap_pen = np.sum(smooth_pen(c_overlap))
        
        # Weights: High weight to ensure validity is prioritized
        w_pen = 50.0
        
        return -np.sum(r_eff) + w_pen * (boundary_pen + overlap_pen)
    
    def optimize_start(x0, y0, r0):
        """
        Run L-BFGS-B optimization from a given start.
        Returns the best valid configuration found.
        """
        params0 = np.concatenate([x0, y0, r0])
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0, 1))  # x
        for i in range(N):
            bounds.append((0, 1))  # y
        for i in range(N):
            bounds.append((0, 0.5))  # r
            
        try:
            res = minimize(objective, params0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            
            # Try to repair by making valid
            x_opt, y_opt, r_opt = make_valid(x_opt, y_opt, r_opt)
            
            # Try to maximize radii further
            x_opt, y_opt, r_opt = maximize_radii(x_opt, y_opt, r_opt)
            
            score = np.sum(r_opt)
            
            return score, np.column_stack((x_opt, y_opt, r_opt))
            
        except Exception:
            return -np.inf, None

    def polish(x, y, r):
        """
        High-precision polish pass.
        Runs L-BFGS-B with tighter tolerances and more iterations.
        """
        params0 = np.concatenate([x, y, r])
        bounds = []
        for i in range(N):
            bounds.append((0, 1))  # x
        for i in range(N):
            bounds.append((0, 1))  # y
        for i in range(N):
            bounds.append((0, 0.5))  # r
            
        try:
            res = minimize(objective, params0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
            
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            
            # Repair and maximize
            x_opt, y_opt, r_opt = make_valid(x_opt, y_opt, r_opt)
            x_opt, y_opt, r_opt = maximize_radii(x_opt, y_opt, r_opt, iterations=100)
            
            score = np.sum(r_opt)
            return score, np.column_stack((x_opt, y_opt, r_opt))
            
        except Exception:
            return np.sum(r), np.column_stack((x, y, r))
    
    def local_search(x, y, r, num_iter=20):
        """
        Improved local search: uses force relaxation and targeted perturbations.
        """
        best_score = np.sum(r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        for _ in range(num_iter):
            if time.time() > deadline - 5:
                break
            
            # Strategy 1: Force relaxation
            if _ % 3 == 0:
                x_new, y_new, r_new = force_relaxation(x, y, r, iterations=50)
            # Strategy 2: Targeted perturbation of bottleneck circles
            elif _ % 3 == 1:
                # Find circles that are limiting others (have many close neighbors)
                x_new = x.copy()
                y_new = y.copy()
                r_new = r.copy()
                
                # Calculate "constraint pressure" for each circle
                pressure = np.zeros(N)
                for i in range(N):
                    for j in range(N):
                        if i == j:
                            continue
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        # Higher pressure when close to limit
                        if dist < r[i] + r[j] + 0.1:
                            pressure[i] += (r[i] + r[j] + 0.1 - dist)
                
                # Perturb high-pressure circles more
                max_pressure = np.max(pressure) if np.max(pressure) > 0 else 1
                scale = 0.03 + 0.07 * (pressure / max_pressure)
                x_new += np.random.normal(0, scale, N)
                y_new += np.random.normal(0, scale, N)
                r_new += np.random.normal(0, scale * 0.3, N)
            # Strategy 3: Random perturbation
            else:
                perturb_scale = 0.04
                x_new = x + np.random.normal(0, perturb_scale, N)
                y_new = y + np.random.normal(0, perturb_scale, N)
                r_new = r + np.random.normal(0, perturb_scale * 0.5, N)
            
            # Clip to bounds
            x_new = np.clip(x_new, 0, 1)
            y_new = np.clip(y_new, 0, 1)
            r_new = np.clip(r_new, 0, 0.5)
            
            # Make valid before optimizing
            x_new, y_new, r_new = make_valid(x_new, y_new, r_new)
            
            # Re-optimize from this new point
            score, config = optimize_start(x_new, y_new, r_new)
            
            if score > best_score:
                best_score = score
                if config is not None:
                    best_x = config[:, 0]
                    best_y = config[:, 1]
                    best_r = config[:, 2]
        
        return best_x, best_y, best_r, best_score
    
    # Initialization strategies
    def init_random():
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.05, 0.15, N)
        return x, y, r
    
    def init_grid():
        # 5x5 grid is 25. Add 1 in center? Or 6x6 and remove 10?
        # Let's use a 5x5 grid for 25 circles, and one in the middle of the square.
        x = []
        y = []
        r = []
        # 5 rows
        for i in range(5):
            y_pos = (i + 0.5) / 5
            if i % 2 == 0:
                # 5 circles
                for j in range(5):
                    x_pos = (j + 0.5) / 5
                    x.append(x_pos)
                    y.append(y_pos)
                    r.append(0.1)
            else:
                # 5 circles, shifted?
                for j in range(5):
                    x_pos = (j + 0.5) / 5
                    x.append(x_pos)
                    y.append(y_pos)
                    r.append(0.1)
        # We have 25 circles. Add 1 more.
        x.append(0.5)
        y.append(0.5)
        r.append(0.1)
        
        # Shuffle to break symmetry
        idx = np.random.permutation(N)
        return np.array(x)[idx], np.array(y)[idx], np.array(r)[idx]
    
    def init_hex():
        # Improved Hexagonal packing - better structured
        x = []
        y = []
        r = []
        
        # Use a hexagonal lattice pattern
        # 6 rows with varying numbers of circles
        row_counts = [5, 5, 5, 4, 4, 3]  # Total: 26
        
        for i in range(6):
            y_pos = (i + 0.5) / 6
            num_circles = row_counts[i]
            
            if i % 2 == 0:
                # Even rows: centered
                for j in range(num_circles):
                    x_pos = (j + 0.5) / (num_circles + 0.5)
                    x.append(x_pos)
                    y.append(y_pos)
                    r.append(0.07)
            else:
                # Odd rows: shifted
                for j in range(num_circles):
                    x_pos = (j + 1.0) / (num_circles + 0.5)
                    x.append(x_pos)
                    y.append(y_pos)
                    r.append(0.07)
        
        # Shuffle to break symmetry
        idx = np.random.permutation(N)
        return np.array(x)[idx], np.array(y)[idx], np.array(r)[idx]
    
    def init_optimized_hex():
        """
        More optimized hexagonal packing with better spacing.
        """
        x = []
        y = []
        r = []
        
        # Try different hexagonal arrangements
        # 6 rows: 5,5,5,4,4,3
        row_counts = [5, 5, 5, 4, 4, 3]
        total_circles = sum(row_counts)
        
        # Calculate optimal spacing
        row_height = 1.0 / len(row_counts)
        
        for i in range(len(row_counts)):
            y_pos = (i + 0.5) * row_height
            num_circles = row_counts[i]
            col_width = 1.0 / (num_circles + 0.5)
            
            if i % 2 == 0:
                for j in range(num_circles):
                    x_pos = (j + 0.5) * col_width
                    x.append(x_pos)
                    y.append(y_pos)
                    # Estimate radius based on local density
                    r.append(0.065)
            else:
                for j in range(num_circles):
                    x_pos = (j + 1.0) * col_width
                    x.append(x_pos)
                    y.append(y_pos)
                    r.append(0.065)
        
        # Shuffle to break symmetry
        idx = np.random.permutation(N)
        return np.array(x)[idx], np.array(y)[idx], np.array(r)[idx]
    
    # Main loop
    num_restarts = 1000 # We will break based on time
    restart = 0
    while time.time() < deadline - 10:
        restart += 1
        # Cycle through initialization strategies
        strategy = restart % 4
        if strategy == 0:
            x0, y0, r0 = init_random()
        elif strategy == 1:
            x0, y0, r0 = init_grid()
        elif strategy == 2:
            x0, y0, r0 = init_hex()
        else:
            x0, y0, r0 = init_optimized_hex()
            
        # Apply force relaxation to get a better starting point
        if time.time() < deadline - 30:
            x0, y0, r0 = force_relaxation(x0, y0, r0, iterations=30)
            x0, y0, r0 = make_valid(x0, y0, r0)
            
        # Run optimization
        score, config = optimize_start(x0, y0, r0)
        
        if score > best_score:
            best_score = score
            if config is not None:
                best_config = config.copy()
                
        # If we found a good score, do some local search to improve it
        if best_config is not None and score > 1.15 and time.time() < deadline - 30:
            x_ls, y_ls, r_ls, score_ls = local_search(best_config[:, 0], best_config[:, 1], best_config[:, 2], num_iter=15)
            if score_ls > best_score:
                best_score = score_ls
                best_config = np.column_stack((x_ls, y_ls, r_ls))

    # Final check
    if best_config is None:
        # Fallback
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.full(N, 0.05)
        return np.column_stack((x, y, r))
    
    # Ensure final validity and polish
    x_f, y_f, r_f = best_config[:, 0], best_config[:, 1], best_config[:, 2]
    x_f, y_f, r_f = make_valid(x_f, y_f, r_f)
    x_f, y_f, r_f = maximize_radii(x_f, y_f, r_f)
    
    # Apply high-precision polish to the best configuration
    if time.time() < deadline - 20:
        score_polished, config_polished = polish(x_f, y_f, r_f)
        if score_polished > np.sum(r_f):
            x_f = config_polished[:, 0]
            y_f = config_polished[:, 1]
            r_f = config_polished[:, 2]
    
    # One more polish pass if time allows
    if time.time() < deadline - 10:
        score_polished2, config_polished2 = polish(x_f, y_f, r_f)
        if score_polished2 > np.sum(r_f):
            x_f = config_polished2[:, 0]
            y_f = config_polished2[:, 1]
            r_f = config_polished2[:, 2]

    final_score = np.sum(r_f)
    
    if final_score < best_score:
        # If make_valid/maximize_radii somehow reduced score (shouldn't happen if logic is correct, but safety first)
        # Re-check original
        if validity_check(best_config[:, 0], best_config[:, 1], best_config[:, 2]):
             final_score = best_score
             x_f, y_f, r_f = best_config[:, 0], best_config[:, 1], best_config[:, 2]
        else:
             # Last resort fallback
             x = np.random.uniform(0.1, 0.9, N)
             y = np.random.uniform(0.1, 0.9, N)
             r = np.full(N, 0.01)
             x_f, y_f, r_f = make_valid(x, y, r)
             x_f, y_f, r_f = maximize_radii(x_f, y_f, r_f)
    
    return np.column_stack((x_f, y_f, r_f))
