"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START

def _smooth_penalty_overlap(x, y, r, i, j):
    """Smooth approximation of max(0, r_i + r_j - dist_ij)^2"""
    d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
    gap = r[i] + r[j] - d
    # Use smooth max: log-sum-exp style, or just softplus
    # softplus(gap) approximates max(0, gap)
    beta = 10.0
    return np.exp(beta * gap) / beta  # This is softplus, but let's use a better one

def _objective(params, w, h, N):
    """
    params: [x_1, y_1, r_1, x_2, y_2, r_2, ..., x_N, y_N, r_N]
    Minimize negative sum of radii + penalties
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Objective: -sum(r) + penalties
    obj = -np.sum(r)
    
    # Penalty for circles going out of bounds
    # For each circle, r_i <= min(x_i, w-x_i, y_i, h-y_i)
    # Use smooth penalty: for each constraint c_i, penalize max(0, r_i - c_i)^2
    eps = 1e-8
    # Left
    gap_l = r - x
    gap_r = r - (w - x)
    gap_b = r - y
    gap_t = r - (h - y)
    
    # Smooth max via softplus
    beta = 20.0
    penalty = 0.0
    for gap in [gap_l, gap_r, gap_b, gap_t]:
        # softplus(gap) = log(1 + exp(beta*gap))/beta approximates max(0, gap)
        # But we want squared, so let's just use softplus for the gap, then square
        soft = np.log1p(np.exp(beta * gap)) / beta
        penalty += np.sum(soft**2)
    
    obj += 10.0 * penalty  # Weight for boundary constraints
    
    # Penalty for overlaps
    # For each pair, penalize max(0, r_i + r_j - dist_ij)^2
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            gap = r[i] + r[j] - d
            soft = np.log1p(np.exp(beta * gap)) / beta
            obj += 10.0 * soft**2
    
    return obj

def _grad(params, w, h, N):
    """Gradient of objective (numerical or analytical). Let's use numerical for simplicity first."""
    # Actually, let's just let L-BFGS-B compute numerical gradient
    return None  # Use numerical

def _solve_for_rect(w, h, N, deadline, initial_configs):
    """Try to solve packing for given rectangle, return best sum of radii and params"""
    best_sum = -np.inf
    best_params = None
    
    for init in initial_configs:
        if time.time() > deadline:
            break
        
        # init: array of shape (N, 3) = [x, y, r]
        params = init.flatten()
        
        # Bounds: x in [r, w-r], y in [r, h-r], r in [0, min(w,h)/2]
        # But bounds depend on r, so use loose bounds and rely on penalty
        # x in [0, w], y in [0, h], r in [0, min(w,h)/2]
        r_max = min(w, h) / 2
        bounds = []
        for i in range(N):
            bounds.append((0, w))      # x_i
            bounds.append((0, h))      # y_i
            bounds.append((0, r_max))  # r_i
        
        # Try L-BFGS-B
        try:
            result = minimize(
                _objective, params, args=(w, h, N),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 50}
            )
            if result.success or result.fun < best_sum:
                x = result.x[0::3]
                y = result.x[1::3]
                r = result.x[2::3]
                # Check if valid
                valid = True
                for i in range(N):
                    if x[i] < -1e-6 or x[i] > w + 1e-6 or y[i] < -1e-6 or y[i] > h + 1e-6:
                        valid = False
                        break
                    if r[i] < -1e-6:
                        valid = False
                        break
                if valid:
                    # Verify no overlaps
                    for i in range(N):
                        for j in range(i+1, N):
                            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                            if d < r[i] + r[j] - 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        s = np.sum(r)
                        if s > best_sum:
                            best_sum = s
                            best_params = result.x.copy()
        except Exception:
            pass
    
    return best_sum, best_params

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    N = 21
    best_sum = 0
    best_result = None
    
    # Try different aspect ratios
    # w + h = 2, so w in (0, 2), h = 2 - w
    # Symmetric, so try w in [0.5, 1.5] (h = 2-w in [0.5, 1.5])
    w_values = np.linspace(0.6, 1.4, 9)
    
    for w in w_values:
        if time.time() > deadline:
            break
        h = 2.0 - w
        
        # Generate initial configurations
        # 1. Grid-like
        # 2. Random with some structure
        initial_configs = []
        
        # Grid 5x5, take first 21
        side = 5
        xs = np.linspace(0.1*w, 0.9*w, side)
        ys = np.linspace(0.1*h, 0.9*h, side)
        grid = []
        for i in range(side):
            for j in range(side):
                if len(grid) < N:
                    grid.append([xs[i], ys[j], min(xs[i], w-xs[i], ys[j], h-ys[j]) * 0.8])
        if len(grid) == N:
            initial_configs.append(np.array(grid))
        
        # Random initializations
        np.random.seed(42)
        for trial in range(3):
            if time.time() > deadline:
                break
            x = np.random.uniform(0.05*w, 0.95*w, N)
            y = np.random.uniform(0.05*h, 0.95*h, N)
            # Initial radii: small
            r = np.full(N, 0.05 * min(w, h))
            init = np.column_stack([x, y, r])
            initial_configs.append(init)
        
        # Run solver
        s, params = _solve_for_rect(w, h, N, deadline, initial_configs)
        
        if s > best_sum and params is not None:
            best_sum = s
            best_result = params.copy()
            # Store w for this result
            best_result_w = w
    
    if best_result is None:
        # Fallback: simple grid
        side = 5
        xs = np.linspace(0.1, 0.9, side)
        ys = np.linspace(0.1, 0.9, side)
        result = []
        for i in range(side):
            for j in range(side):
                if len(result) < N:
                    x = xs[i]
                    y = ys[j]
                    r = 0.5 / side - 0.01
                    result.append([x, y, r])
        return np.array(result)
    
    # Extract x, y, r from best_result
    x = best_result[0::3]
    y = best_result[1::3]
    r = best_result[2::3]
    
    # Ensure valid: clip to rectangle
    w = best_result_w
    h = 2.0 - w
    for i in range(N):
        x[i] = np.clip(x[i], 0, w)
        y[i] = np.clip(y[i], 0, h)
        r[i] = np.clip(r[i], 0, min(w, h)/2)
        # Ensure fits in bounds
        r[i] = min(r[i], x[i], w-x[i], y[i], h-y[i])
    
    result = np.column_stack([x, y, r])
    return result

# EVOLVE-BLOCK-END
