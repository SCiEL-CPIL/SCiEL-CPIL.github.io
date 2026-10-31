"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

# EVOLVE-BLOCK-START

def construct():
    """
    Pack 21 circles in a variable rectangle of perimeter 4.
    Returns array of shape (21, 3): [cx, cy, r] for each circle.
    Uses force-directed relaxation with SLSQP optimization on aspect ratio and positions.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial aspect ratio guess: square is 1x1 (perimeter 4)
    # We will optimize aspect ratio w/h and positions
    
    # Initial configuration: grid-like
    side = int(np.ceil(np.sqrt(N)))  # 5x5 grid for 21
    w0, h0 = 1.0, 1.0  # Start with square
    
    # Generate initial positions in unit square, will be scaled
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                positions.append([0.1 + 0.8 * i / (side - 1), 0.1 + 0.8 * j / (side - 1)])
    positions = np.array(positions[:N])
    
    # Initial radii: small
    r0 = 0.05 * np.ones(N)
    
    best_value = 0.0
    best_result = None
    
    # Objective function: maximize sum of radii subject to constraints
    # We parameterize by: aspect_ratio (log), positions (x, y in [0,1]x[0,1]), radii (positive)
    # Rectangle dimensions: w = 2 * exp(ar), h = 2 * exp(-ar) so perimeter = 2(w+h) = 4
    # Actually: perimeter = 2(w + h) = 4 => w + h = 2
    # Let w = 1 + exp(ar), h = 1 - exp(ar) ... no, need w,h > 0
    # Better: w = 2 / (1 + exp(-ar)), h = 2 - w = 2 * exp(-ar) / (1 + exp(-ar))
    # Or simpler: w = 1 + 0.5 * np.tanh(ar) * 2 ... let's use:
    # w = 2 / (1 + np.exp(-ar)), h = 2 * np.exp(-ar) / (1 + np.exp(-ar))
    # This ensures w + h = 2, w,h > 0
    
    def get_dims(ar):
        w = 2.0 / (1.0 + np.exp(-ar))
        h = 2.0 * np.exp(-ar) / (1.0 + np.exp(-ar))
        return w, h
    
    def objective(params):
        """Minimize negative sum of radii (to maximize)."""
        n_params = 1 + 2 * N + N  # 1 aspect + 2N positions + N radii
        ar = params[0]
        pos = params[1:1 + 2 * N].reshape(N, 2)  # normalized [0,1]x[0,1]
        radii = params[1 + 2 * N:]
        
        w, h = get_dims(ar)
        
        # Actual positions
        cx = pos[:, 0] * w
        cy = pos[:, 1] * h
        
        # Sum of radii
        sum_r = np.sum(radii)
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        # Penalty for circles going outside rectangle
        for i in range(N):
            # Left/right walls
            if cx[i] < radii[i]:
                penalty += (radii[i] - cx[i]) ** 2
            if cx[i] > w - radii[i]:
                penalty += (cx[i] - (w - radii[i])) ** 2
            # Top/bottom walls
            if cy[i] < radii[i]:
                penalty += (radii[i] - cy[i]) ** 2
            if cy[i] > h - radii[i]:
                penalty += (cy[i] - (h - radii[i])) ** 2
        
        # Penalty for negative radii
        neg_r = np.sum(np.maximum(0, -radii) ** 2)
        
        return -sum_r + 100.0 * penalty + 10.0 * neg_r
    
    # Try multiple random restarts
    n_restarts = 5
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Randomize initial positions
        if restart == 0:
            pos_init = positions.copy()
            ar_init = 0.0
            r_init = r0.copy()
        else:
            pos_init = 0.2 + 0.6 * rng.random((N, 2))
            ar_init = rng.uniform(-1.5, 1.5)
            r_init = 0.03 + 0.05 * rng.random(N)
        
        params0 = np.concatenate([[ar_init], pos_init.flatten(), r_init])
        
        # Bounds
        bounds = [(-3.0, 3.0)]  # aspect ratio
        for _ in range(N):
            bounds.append((0.0, 1.0))  # x position
            bounds.append((0.0, 1.0))  # y position
        for _ in range(N):
            bounds.append((0.001, 0.5))  # radius
        
        try:
            result = minimize(
                objective,
                params0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if time.time() > deadline:
                break
            
            # Extract best solution
            ar_best = result.x[0]
            pos_best = result.x[1:1 + 2 * N].reshape(N, 2)
            r_best = result.x[1 + 2 * N:]
            
            w_best, h_best = get_dims(ar_best)
            cx_best = pos_best[:, 0] * w_best
            cy_best = pos_best[:, 1] * h_best
            
            # Verify validity
            valid = True
            for i in range(N):
                if r_best[i] < 1e-6:
                    valid = False
                    break
                if cx_best[i] < r_best[i] - 1e-6 or cx_best[i] > w_best - r_best[i] + 1e-6:
                    valid = False
                    break
                if cy_best[i] < r_best[i] - 1e-6 or cy_best[i] > h_best - r_best[i] + 1e-6:
                    valid = False
                    break
            
            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = cx_best[i] - cx_best[j]
                        dy = cy_best[i] - cy_best[j]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist < r_best[i] + r_best[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                value = np.sum(r_best)
                if value > best_value:
                    best_value = value
                    result_arr = np.column_stack([cx_best, cy_best, r_best])
                    best_result = result_arr
                    
        except Exception as e:
            pass
    
    # If no valid result found, fall back to seed
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result

# EVOLVE-BLOCK-END
