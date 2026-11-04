import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    # Time limit check
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    def penalty_function(params):
        # params: (N, 3) array of [x, y, r]
        x = params[:, 0]
        y = params[:, 1]
        r = params[:, 2]
        
        # Sum of radii (to be maximized, so we minimize -sum)
        score = -np.sum(r)
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    penalty += overlap * overlap
        
        # Penalty for walls
        wall_penalty = 0.0
        for i in range(N):
            # Distance to walls
            d_walls = np.array([x[i], 1 - x[i], y[i], 1 - y[i]])
            min_d = np.min(d_walls)
            wall_viol = r[i] - min_d
            if wall_viol > 0:
                wall_penalty += wall_viol * wall_viol
        
        # Also penalize negative radius
        neg_r = np.sum(np.maximum(0, -r) ** 2)
        
        # Weights
        w_overlap = 100.0
        w_wall = 100.0
        
        return score + w_overlap * penalty + w_wall * wall_penalty + 10.0 * neg_r

    def penalty_gradient(params):
        # Analytical gradient for speed
        x = params[:, 0]
        y = params[:, 1]
        r = params[:, 2]
        
        grad = np.zeros_like(params)
        
        # Gradient of -sum(r)
        grad[:, 2] -= 1.0
        
        # Gradient of overlap penalty
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-6:
                    dist = 1e-6
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    # d/dx_i (overlap^2) = 2 * overlap * d(overlap)/dx_i
                    # d(overlap)/dx_i = -dx/dist
                    # d(overlap)/dx_j = dx/dist
                    # d(overlap)/dr_i = 1
                    # d(overlap)/dr_j = 1
                    
                    factor = 2.0 * overlap
                    grad[i, 0] += 100.0 * factor * (-dx / dist)
                    grad[j, 0] += 100.0 * factor * (dx / dist)
                    grad[i, 1] += 100.0 * factor * (-dy / dist)
                    grad[j, 1] += 100.0 * factor * (dy / dist)
                    grad[i, 2] += 100.0 * factor * 1.0
                    grad[j, 2] += 100.0 * factor * 1.0
        
        # Gradient of wall penalty
        for i in range(N):
            d_walls = np.array([x[i], 1 - x[i], y[i], 1 - y[i]])
            min_d = np.min(d_walls)
            wall_viol = r[i] - min_d
            if wall_viol > 0:
                factor = 2.0 * wall_viol
                # Which wall is the minimum?
                # If x[i] is min, d/dx_i of (r - x)^2 = 2(r-x)*(-1) = -2 wall_viol
                # If 1-x[i] is min, d/dx_i of (r - (1-x))^2 = 2(r - (1-x))*(1) = 2 wall_viol
                # We need to be careful with the gradient of min.
                # Simpler: just push away from the closest wall.
                # Actually, let's just use a soft repulsion from all 4 walls to avoid corner cases in gradient.
                # Or just calculate gradient of the specific violating wall.
                
                # Let's re-implement wall penalty as sum of squares of violations for each wall independently?
                # No, that's too strong. 
                # Let's stick to the min wall but handle gradient carefully.
                # If min_d is x[i], grad_x = -2*wall_viol. grad_r = 2*wall_viol.
                # If min_d is 1-x[i], grad_x = 2*wall_viol. grad_r = 2*wall_viol.
                
                if d_walls[0] == min_d:
                    grad[i, 0] += 100.0 * factor * (-1.0)
                    grad[i, 2] += 100.0 * factor * 1.0
                elif d_walls[1] == min_d:
                    grad[i, 0] += 100.0 * factor * (1.0)
                    grad[i, 2] += 100.0 * factor * 1.0
                elif d_walls[2] == min_d:
                    grad[i, 1] += 100.0 * factor * (-1.0)
                    grad[i, 2] += 100.0 * factor * 1.0
                elif d_walls[3] == min_d:
                    grad[i, 1] += 100.0 * factor * (1.0)
                    grad[i, 2] += 100.0 * factor * 1.0

        # Gradient of neg_r penalty
        neg_r_mask = r < 0
        grad[neg_r_mask, 2] += 20.0 * (-r[neg_r_mask])
        
        return grad

    def run_optimization(initial_params):
        # Use L-BFGS-B
        res = minimize(
            penalty_function, 
            initial_params, 
            method='L-BFGS-B', 
            jac=penalty_gradient,
            options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-5}
        )
        return res.x

    # Initialization: Hexagonal grid
    # Generate a hex grid of points
    hex_xs = []
    hex_ys = []
    s = 0.2
    h = s * np.sqrt(3) / 2
    for i in range(8):
        for j in range(8):
            x = 0.5 + (j - 3.5) * s + (i % 2) * s / 2
            y = 0.5 + (i - 3.5) * h
            if 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95:
                hex_xs.append(x)
                hex_ys.append(y)
    
    # Select 26 points closest to the center or random sample
    # Let's just take the first 26 from a shuffled list to ensure variety
    np.random.seed(42)
    if len(hex_xs) >= N:
        indices = np.random.permutation(len(hex_xs))[:N]
        xs = np.array(hex_xs)[indices]
        ys = np.array(hex_ys)[indices]
    else:
        xs = np.random.rand(N) * 0.8 + 0.1
        ys = np.random.rand(N) * 0.8 + 0.1
        
    rs = np.full(N, 0.02)
    
    initial_params = np.column_stack((xs, ys, rs))
    
    best_params = None
    best_score = -np.inf
    
    # Multi-start
    for k in range(5):
        if time.time() - start_time > time_limit:
            break
            
        # Perturb initial params for multi-start
        if k > 0:
            current_params = initial_params + np.random.normal(0, 0.05, initial_params.shape)
            # Clamp positions
            current_params[:, 0] = np.clip(current_params[:, 0], 0.05, 0.95)
            current_params[:, 1] = np.clip(current_params[:, 1], 0.05, 0.95)
            current_params[:, 2] = np.clip(current_params[:, 2], 0.01, 0.2)
        else:
            current_params = initial_params.copy()
            
        res_params = run_optimization(current_params)
        
        # Evaluate raw score (sum of radii) if valid
        x = res_params[:, 0]
        y = res_params[:, 1]
        r = res_params[:, 2]
        
        # Check validity
        valid = True
        for i in range(N):
            if r[i] > x[i] + 1e-9 or r[i] > 1 - x[i] + 1e-9 or r[i] > y[i] + 1e-9 or r[i] > 1 - y[i] + 1e-9:
                valid = False
                break
            if r[i] < 0:
                valid = False
                break
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r[i] + r[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
                    
        if valid:
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_params = res_params.copy()
                
        # If invalid, try to fix by shrinking radii
        else:
            # Shrink radii to make it valid
            scale = 1.0
            for _ in range(100):
                x = res_params[:, 0]
                y = res_params[:, 1]
                r = res_params[:, 2] * scale
                valid = True
                for i in range(N):
                    if r[i] > x[i] + 1e-9 or r[i] > 1 - x[i] + 1e-9 or r[i] > y[i] + 1e-9 or r[i] > 1 - y[i] + 1e-9:
                        valid = False
                        break
                    if r[i] < 0:
                        valid = False
                        break
                if valid:
                    for i in range(N):
                        for j in range(i + 1, N):
                            dx = x[i] - x[j]
                            dy = y[i] - y[j]
                            dist = np.sqrt(dx*dx + dy*dy)
                            if dist < r[i] + r[j] - 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                if valid:
                    score = np.sum(r)
                    if score > best_score:
                        best_score = score
                        best_params = np.column_stack((x, y, r))
                    break
                scale *= 0.9

    if best_params is None:
        # Fallback to initial
        best_params = initial_params
        
    # Final cleanup: ensure strict validity
    x = best_params[:, 0]
    y = best_params[:, 1]
    r = best_params[:, 2]
    
    # Shrink radii slightly to ensure no floating point errors
    r *= 0.999
    
    result = np.zeros((N, 3))
    result[:, 0] = x
    result[:, 1] = y
    result[:, 2] = r
    
    return result
