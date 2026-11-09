"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def construct():
    start_time = time.time()
    time_limit = 135.0  # Safety margin for 150s limit
    
    best_sum = 0.0
    best_config = None
    best_W = None
    best_H = None
    
    def validate(config, W, H, tol=1e-7):
        """Check if configuration is valid and return sum of radii."""
        if config.shape != (N, 3):
            return 0.0, False
        for i in range(N):
            x, y, r = config[i]
            if r < -tol:
                return 0.0, False
            r = max(r, 0.0)
            if x < r - tol or x > W - r + tol:
                return 0.0, False
            if y < r - tol or y > H - r + tol:
                return 0.0, False
        for i in range(N):
            for j in range(i + 1, N):
                x1, y1, r1 = config[i]
                x2, y2, r2 = config[j]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < (r1 + r2) - tol:
                    return 0.0, False
        s = np.sum(config[:, 2])
        return s, True
    
    def objective(vars, W, H):
        """Penalized objective: minimize -sum(r) + penalties."""
        config = vars.reshape(N, 3)
        penalty = 0.0
        
        # Boundary penalties
        for i in range(N):
            x, y, r = config[i]
            if r < 0:
                penalty += (-r) * 50.0
            if x < r:
                penalty += (r - x) * 20.0
            if x > W - r:
                penalty += (x - (W - r)) * 20.0
            if y < r:
                penalty += (r - y) * 20.0
            if y > H - r:
                penalty += (y - (H - r)) * 20.0
        
        # Overlap penalties (smooth)
        for i in range(N):
            for j in range(i + 1, N):
                x1, y1, r1 = config[i]
                x2, y2, r2 = config[j]
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                min_dist = r1 + r2
                if dist < min_dist:
                    penalty += (min_dist - dist) * 15.0
        
        sum_r = np.sum(config[:, 2])
        return -sum_r + penalty
    
    def grid_init(W, H, n_cols, n_rows):
        """Create a grid-based initial configuration."""
        config = np.zeros((N, 3))
        idx = 0
        for j in range(n_rows):
            for i in range(n_cols):
                if idx >= N:
                    break
                x = W * (i + 0.5) / n_cols
                y = H * (j + 0.5) / n_rows
                # Conservative radius
                dx = W / n_cols
                dy = H / n_rows
                r = 0.4 * min(dx, dy)
                config[idx] = [x, y, r]
                idx += 1
            if idx >= N:
                break
        # Fill remaining with small circles
        while idx < N:
            config[idx] = [W / 2, H / 2, 0.01]
            idx += 1
        return config
    
    def relax(config, W, H, max_iter=500, lr_init=0.05, lr_decay=0.995):
        """Iterative relaxation with gradient descent."""
        config = config.copy().astype(float)
        lr = lr_init
        prev_obj = objective(config.flatten(), W, H)
        
        for it in range(max_iter):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Compute gradient numerically (finite differences)
            grad = np.zeros_like(config.flatten())
            eps = 1e-5
            for k in range(len(config.flatten())):
                old = config.flatten()[k]
                config.flatten()[k] = old + eps
                f_plus = objective(config.flatten(), W, H)
                config.flatten()[k] = old - eps
                f_minus = objective(config.flatten(), W, H)
                config.flatten()[k] = old
                grad[k] = (f_plus - f_minus) / (2 * eps)
            
            # Update
            config = (config.flatten() - lr * grad).reshape(N, 3)
            
            # Clip radii to be non-negative
            config[:, 2] = np.maximum(config[:, 2], 0.0)
            
            new_obj = objective(config.flatten(), W, H)
            if abs(prev_obj - new_obj) < 1e-8:
                break
            prev_obj = new_obj
            lr *= lr_decay
        
        return config
    
    def scipy_polish(config, W, H):
        """Use scipy minimize to polish the configuration."""
        x0 = config.flatten()
        try:
            res = minimize(objective, x0, args=(W, H), method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            config = res.x.reshape(N, 3)
            config[:, 2] = np.maximum(config[:, 2], 0.0)
            return config
        except Exception:
            return config
    
    # Try multiple aspect ratios and grid configurations
    # Perimeter = 2*(W+H) = 4, so W+H = 2
    aspect_ratios = np.linspace(0.4, 2.5, 30)
    
    for ar in aspect_ratios:
        if time.time() - start_time > time_limit - 10:
            break
        
        W = 2.0 * ar / (1 + ar)
        H = 2.0 - W
        if W <= 0.1 or H <= 0.1:
            continue
        
        # Try different grid layouts
        grid_configs = [
            (5, 5),  # 25 spots
            (6, 4),  # 24 spots
            (4, 6),  # 24 spots
            (7, 3),  # 21 spots
            (3, 7),  # 21 spots
            (7, 4),  # 28 spots
            (4, 7),  # 28 spots
            (6, 5),  # 30 spots
            (5, 6),  # 30 spots
            (8, 3),  # 24 spots
            (3, 8),  # 24 spots
        ]
        
        # Filter grids that fit the aspect ratio reasonably
        for n_cols, n_rows in grid_configs:
            if time.time() - start_time > time_limit - 5:
                break
            
            grid_ar = n_cols / n_rows
            if abs(grid_ar - W / H) / (W / H) > 0.8:
                continue
            
            init_config = grid_init(W, H, n_cols, n_rows)
            
            # Relax
            relaxed = relax(init_config, W, H, max_iter=200, lr_init=0.03)
            
            # Polish with scipy
            polished = scipy_polish(relaxed, W, H)
            
            s, valid = validate(polished, W, H)
            if valid and s > best_sum:
                best_sum = s
                best_config = polished.copy()
                best_W = W
                best_H = H
    
    # Final polish on the best configuration found
    if best_config is not None and time.time() - start_time < time_limit - 10:
        final = scipy_polish(best_config, best_W, best_H)
        s, valid = validate(final, best_W, best_H)
        if valid and s > best_sum:
            best_sum = s
            best_config = final.copy()
    
    if best_config is None:
        # Fallback: uniform grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for j in range(side):
            for i in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.005
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_config
# EVOLVE-BLOCK-END
