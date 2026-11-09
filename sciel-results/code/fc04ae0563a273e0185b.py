import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin
    
    best_score = 0.0
    best_config = None
    
    def objective(params):
        """
        Objective function for L-BFGS-B.
        We want to maximize sum(r), so we minimize -sum(r).
        We add penalties for boundary violations and circle overlaps.
        """
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Ensure radii are positive (L-BFGS-B handles bounds, but we clip for safety in penalty calc)
        r = np.maximum(r, 0)
        
        # 1. Boundary Penalties
        # If x < r, circle sticks out left. Penalty: (r - x)^2
        # If x > 1-r, circle sticks out right. Penalty: (x - (1-r))^2
        # Same for y.
        pen_x_left = np.maximum(0, r - x)**2
        pen_x_right = np.maximum(0, x - (1 - r))**2
        pen_y_bottom = np.maximum(0, r - y)**2
        pen_y_top = np.maximum(0, y - (1 - r))**2
        boundary_pen = np.sum(pen_x_left + pen_x_right + pen_y_bottom + pen_y_top)
        
        # 2. Overlap Penalties
        # For each pair i, j: if dist < r_i + r_j, penalty (r_i + r_j - dist)^2
        overlap_pen = 0.0
        # Vectorized distance calculation is faster for small N, but loop is fine for N=26
        # Let's use a loop for clarity and to avoid constructing large matrices if not needed, 
        # though for N=26, a 26x26 matrix is trivial.
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                overlap = r[i] + r[j] - d
                if overlap > 0:
                    overlap_pen += overlap**2
        
        # Weights: Need to balance maximizing radius vs minimizing penalties.
        # If penalty is too low, circles overlap. If too high, radii stay small.
        # Adaptive weight or fixed high weight? 
        # Let's use a high weight for penalties to ensure validity is prioritized initially, 
        # then we can rely on the fact that if penalties are 0, we are maximizing sum(r).
        w_pen = 100.0
        
        return -np.sum(r) + w_pen * (boundary_pen + overlap_pen)

    def get_valid_config(x, y, r):
        """
        Check if configuration is valid and return sum of radii if so, else -inf.
        """
        if np.any(r < 0):
            return -np.inf
        if np.any(x - r < -1e-9) or np.any(y - r < -1e-9):
            return -np.inf
        if np.any(x + r > 1 + 1e-9) or np.any(y + r > 1 + 1e-9):
            return -np.inf
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < r[i] + r[j] - 1e-9:
                    return -np.inf
        return np.sum(r)

    def optimize_start(x0, y0, r0):
        """
        Run L-BFGS-B optimization from a given start.
        Returns the best valid configuration found.
        """
        params0 = np.concatenate([x0, y0, r0])
        
        # Bounds
        # x, y in [0, 1]
        # r in [0, 0.5] (max possible radius in unit square is 0.5)
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
            
            # The optimizer might end up in a state with low penalty but not zero.
            # We need to check validity.
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:3*N]
            
            # Sometimes L-BFGS-B leaves tiny overlaps. We can try to "repair" by shrinking radii slightly.
            # But first, check if it's valid.
            score = get_valid_config(x_opt, y_opt, r_opt)
            
            if score == -np.inf:
                # Try to repair by scaling down radii
                scale = 1.0
                for _ in range(10):
                    r_scaled = r_opt * scale
                    if get_valid_config(x_opt, y_opt, r_scaled) != -np.inf:
                        score = np.sum(r_scaled)
                        break
                    scale *= 0.95
                if score == -np.inf:
                    return -np.inf, None
            else:
                # It's valid. Can we increase radii further?
                # The optimizer maximized -sum(r) + penalty. If penalty is 0, it should be near optimal.
                # But we can try a local search to see if we can push radii up more.
                pass
                
            return score, np.column_stack((x_opt, y_opt, r_opt))
            
        except Exception:
            return -np.inf, None

    def local_search(x, y, r, num_iter=20):
        """
        Simple local search: perturb positions/radii, re-optimize.
        """
        best_score = get_valid_config(x, y, r)
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        for _ in range(num_iter):
            if time.time() > deadline - 5:
                break
            
            # Perturb
            perturb_scale = 0.05
            x_new = x + np.random.normal(0, perturb_scale, N)
            y_new = y + np.random.normal(0, perturb_scale, N)
            r_new = r + np.random.normal(0, perturb_scale * 0.5, N)
            
            # Clip to bounds
            x_new = np.clip(x_new, 0, 1)
            y_new = np.clip(y_new, 0, 1)
            r_new = np.clip(r_new, 0, 0.5)
            
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
        # Actually, a hexagonal packing is better.
        # But for simplicity, random is often better for multi-start.
        # Let's try a structured start: 5 rows, 5-6 cols.
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
        # Hexagonal packing approximation
        x = []
        y = []
        r = []
        # Radius approx 0.1
        # Rows
        for i in range(6):
            y_pos = (i + 0.5) / 6
            if i % 2 == 0:
                # 5 circles
                for j in range(5):
                    x_pos = (j + 0.5) / 5
                    x.append(x_pos)
                    y.append(y_pos)
                    r.append(0.1)
            else:
                # 4 circles, shifted
                for j in range(4):
                    x_pos = (j + 1.0) / 5
                    x.append(x_pos)
                    y.append(y_pos)
                    r.append(0.1)
        # We have 5*3 + 4*3 = 15 + 12 = 27 circles. Need 26.
        # Remove one random.
        idx = np.random.choice(N, N-1, replace=False) # This is wrong, N is 26, we have 27 points.
        # Let's just take the first 26.
        x = x[:26]
        y = y[:26]
        r = r[:26]
        idx = np.random.permutation(26)
        return np.array(x)[idx], np.array(y)[idx], np.array(r)[idx]

    # Main loop
    num_restarts = 1000 # We will break based on time
    restart = 0
    while time.time() < deadline - 10:
        restart += 1
        if restart % 3 == 1:
            x0, y0, r0 = init_random()
        elif restart % 3 == 2:
            x0, y0, r0 = init_grid()
        else:
            x0, y0, r0 = init_hex()
            
        # Run optimization
        score, config = optimize_start(x0, y0, r0)
        
        if score > best_score:
            best_score = score
            if config is not None:
                best_config = config.copy()
                
        # If we found a good score, do some local search to improve it
        if best_config is not None and score > 1.1 and time.time() < deadline - 30:
            x_ls, y_ls, r_ls, score_ls = local_search(best_config[:, 0], best_config[:, 1], best_config[:, 2], num_iter=10)
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
    
    # Ensure final validity
    final_score = get_valid_config(best_config[:, 0], best_config[:, 1], best_config[:, 2])
    if final_score == -np.inf:
        # Shrink radii until valid
        x, y, r = best_config[:, 0], best_config[:, 1], best_config[:, 2]
        scale = 1.0
        for _ in range(50):
            r_new = r * scale
            if get_valid_config(x, y, r_new) != -np.inf:
                best_config = np.column_stack((x, y, r_new))
                break
            scale *= 0.95
        else:
            # Extreme fallback
            x = np.random.uniform(0.1, 0.9, N)
            y = np.random.uniform(0.1, 0.9, N)
            r = np.full(N, 0.01)
            best_config = np.column_stack((x, y, r))

    return best_config
