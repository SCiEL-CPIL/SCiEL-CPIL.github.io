"""Circle packing in perimeter-4 rectangle: gradient-based local search with radius expansion."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    best_sum_r = -np.inf
    best_result = None
    
    def get_elapsed():
        return time.time() - start_time
    
    def is_time_up():
        return time.time() > deadline
    
    def check_validity(x, y, r, w, h, tol=1e-7):
        """Check if the configuration is valid."""
        if w <= 0 or h <= 0 or w + h != 2.0:
            return False
        for i in range(N):
            if r[i] < -tol:
                return False
            if x[i] < -tol or x[i] > w - r[i] + tol:
                return False
            if y[i] < -tol or y[i] > h - r[i] + tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                min_d = r[i] + r[j]
                if d2 < min_d * min_d - tol:
                    return False
        return True
    
    def expand_radii(x, y, r, w, h, max_iter=200):
        """Try to expand radii while maintaining validity."""
        r = r.copy()
        improved = True
        while improved and not is_time_up():
            improved = False
            for i in range(N):
                # Try to increase radius of circle i
                # Find the maximum possible radius given current positions
                max_r = min(w - x[i], x[i], h - y[i], y[i])
                # Check against all other circles
                for j in range(N):
                    if i == j:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    # We need d >= r[i] + r[j], so r[i] <= d - r[j]
                    available = d - r[j]
                    if available < max_r:
                        max_r = available
                max_r = max(0, max_r)
                
                # Try to increase to max_r (or a fraction if we're being conservative)
                target_r = max_r
                if target_r > r[i] * 1.001:
                    # Try the full expansion
                    old_r = r[i]
                    r[i] = target_r
                    if check_validity(x, y, r, w, h, tol=1e-6):
                        improved = True
                    else:
                        r[i] = old_r
        return r
    
    def local_search_positions(x, y, r, w, h, max_iter=500):
        """Local search to improve positions, allowing for larger radii."""
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        for iteration in range(max_iter):
            if is_time_up():
                break
            
            improved = False
            # Try small random perturbations
            for i in range(N):
                # Try moving circle i slightly
                for dx in [-0.01, 0.01, -0.005, 0.005]:
                    for dy in [-0.01, 0.01, -0.005, 0.005]:
                        old_x, old_y = x[i], y[i]
                        x[i] += dx
                        y[i] += dy
                        
                        # Check if still valid
                        if check_validity(x, y, r, w, h, tol=1e-6):
                            # Try to expand radii after this move
                            new_r = expand_radii(x, y, r, w, h, max_iter=50)
                            if np.sum(new_r) > np.sum(r) + 1e-8:
                                r = new_r
                                improved = True
                            else:
                                x[i] = old_x
                                y[i] = old_y
                        else:
                            x[i] = old_x
                            y[i] = old_y
            
            if not improved:
                break
        
        return x, y, r
    
    def grid_init(w, h):
        """Initialize circles in a grid pattern."""
        # Determine grid dimensions
        aspect = w / h
        if aspect >= 1:
            cols = int(np.ceil(np.sqrt(N * aspect)))
            rows = int(np.ceil(N / cols))
        else:
            rows = int(np.ceil(np.sqrt(N / aspect)))
            cols = int(np.ceil(N / rows))
        
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        cell_w = w / cols
        cell_h = h / rows
        
        idx = 0
        for i in range(cols):
            for j in range(rows):
                if idx >= N:
                    break
                x[idx] = (i + 0.5) * cell_w
                y[idx] = (j + 0.5) * cell_h
                r[idx] = 0.5 * min(cell_w, cell_h) * 0.8
                idx += 1
        
        return x, y, r
    
    def optimize_with_slsqp(x0, y0, r0, w0):
        """Use SLSQP to optimize the configuration."""
        h0 = 2.0 - w0
        z0 = np.concatenate([x0, y0, r0, [w0]])
        
        def objective(z):
            x = z[0:N]
            y = z[N:2*N]
            r = z[2*N:3*N]
            w = z[3*N]
            h = 2.0 - w
            
            if w <= 0.05 or h <= 0.05:
                return 1e10
            
            # Penalty for invalid configurations
            pen = 0.0
            # Boundary penalties
            pen += np.sum(np.maximum(0, -x)**2)
            pen += np.sum(np.maximum(0, x - (w - r))**2)
            pen += np.sum(np.maximum(0, -y)**2)
            pen += np.sum(np.maximum(0, y - (h - r))**2)
            
            # Overlap penalties
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d2 = dx * dx + dy * dy
                    min_d2 = (r[i] + r[j])**2
                    if d2 < min_d2:
                        diff = min_d2 - d2
                        pen += diff * diff
            
            return -np.sum(r) + 50.0 * pen
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0.0, 2.0))  # x
        for i in range(N):
            bounds.append((0.0, 2.0))  # y
        for i in range(N):
            bounds.append((0.0, 1.0))  # r
        bounds.append((0.1, 1.9))  # w
        
        try:
            res = minimize(objective, z0, method='SLSQP', bounds=bounds,
                          options={'maxiter': 100, 'ftol': 1e-5})
            
            if res.fun < 1e8:  # Not completely invalid
                x = res.x[0:N]
                y = res.x[N:2*N]
                r = res.x[2*N:3*N]
                w = res.x[3*N]
                h = 2.0 - w
                
                # Clip to valid range
                r = np.maximum(r, 0)
                x = np.clip(x, 0, w - r)
                y = np.clip(y, 0, h - r)
                
                return x, y, r, w, h
        except Exception:
            pass
        
        return x0, y0, r0, w0, h0
    
    # Try multiple initializations and aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.43, 0.6, 1.67]
    
    for aspect in aspect_ratios:
        if is_time_up():
            break
        
        w = 2.0 * aspect / (1 + aspect)
        h = 2.0 - w
        
        x0, y0, r0 = grid_init(w, h)
        
        # Optimize with SLSQP
        x, y, r, w_opt, h_opt = optimize_with_slsqp(x0, y0, r0, w)
        
        if not check_validity(x, y, r, w_opt, h_opt, tol=1e-5):
            # Try to fix by shrinking radii
            for _ in range(10):
                r = r * 0.95
                if check_validity(x, y, r, w_opt, h_opt, tol=1e-5):
                    break
        
        if check_validity(x, y, r, w_opt, h_opt, tol=1e-5):
            # Expand radii
            r_expanded = expand_radii(x, y, r, w_opt, h_opt)
            sum_r = np.sum(r_expanded)
            
            if sum_r > best_sum_r:
                best_sum_r = sum_r
                best_result = (x.copy(), y.copy(), r_expanded.copy(), w_opt, h_opt)
        
        # Local search
        if not is_time_up() and check_validity(x, y, r, w_opt, h_opt, tol=1e-5):
            x_ls, y_ls, r_ls = local_search_positions(x, y, r, w_opt, h_opt, max_iter=100)
            if check_validity(x_ls, y_ls, r_ls, w_opt, h_opt, tol=1e-5):
                sum_r_ls = np.sum(r_ls)
                if sum_r_ls > best_sum_r:
                    best_sum_r = sum_r_ls
                    best_result = (x_ls.copy(), y_ls.copy(), r_ls.copy(), w_opt, h_opt)
    
    # If no valid solution found, return a trivial one
    if best_result is None:
        x, y, r = grid_init(1.0, 1.0)
        return np.column_stack([x, y, r])
    
    x, y, r, w, h = best_result
    
    # Final validation and cleanup
    r = np.maximum(r, 0)
    
    return np.column_stack([x, y, r])

# EVOLVE-BLOCK-END
