"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START

def _penalty(pos, radii, n, lambda_val):
    """
    Calculate the penalty term for constraint violations.
    Constraints:
    1. Circles must be inside the unit square: x_i - r_i >= 0, y_i - r_i >= 0, x_i + r_i <= 1, y_i + r_i <= 1
    2. Circles must not overlap: dist(i, j) >= r_i + r_j
    """
    penalty = 0.0
    
    # Boundary constraints
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        # Violation if circle goes out of bounds
        v1 = max(0, r - x)
        v2 = max(0, r - y)
        v3 = max(0, r - (1 - x))
        v4 = max(0, r - (1 - y))
        penalty += (v1**2 + v2**2 + v3**2 + v4**2)
        
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            r_sum = radii[i] + radii[j]
            v = max(0, r_sum - dist)
            penalty += v**2
            
    return lambda_val * penalty

def _objective(z, n, lambda_val):
    """
    Objective function to minimize.
    z is a vector of length 3n: [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    We maximize sum of radii, so we minimize -sum(radii) + penalty.
    """
    pos = z[:2*n].reshape(n, 2)
    radii = z[2*n:]
    
    # Ensure radii are non-negative in the objective evaluation (though bounds handle it)
    radii = np.maximum(radii, 0)
    
    sum_radii = np.sum(radii)
    pen = _penalty(pos, radii, n, lambda_val)
    
    return -sum_radii + pen

def _is_valid(pos, radii, n, tol=1e-5):
    """Check if the packing is valid."""
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        if r < -tol: return False
        if x - r < -tol or y - r < -tol or x + r > 1 + tol or y + r > 1 + tol:
            return False
            
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True

def _fix_and_validate(pos, radii, n):
    """
    Attempt to fix a potentially invalid solution by clamping radii.
    Returns (pos, radii, score) if valid, else (None, None, -inf).
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # Clamp radii to be non-negative
    radii = np.maximum(radii, 0)
    
    # Clamp radii to boundary constraints
    for i in range(n):
        x, y = pos[i]
        max_r = min(x, y, 1 - x, 1 - y)
        if radii[i] > max_r:
            radii[i] = max(0, max_r)
            
    # Iteratively reduce radii to fix overlaps
    for _ in range(50):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                r_sum = radii[i] + radii[j]
                if dist < r_sum - 1e-8:
                    excess = r_sum - dist
                    # Reduce both radii proportionally or equally
                    reduction = excess / 2.0 + 1e-6
                    radii[i] = max(0, radii[i] - reduction)
                    radii[j] = max(0, radii[j] - reduction)
                    changed = True
        if not changed:
            break
            
    # Final boundary check
    for i in range(n):
        x, y = pos[i]
        max_r = min(x, y, 1 - x, 1 - y)
        if radii[i] > max_r:
            radii[i] = max(0, max_r)

    if _is_valid(pos, radii, n):
        return pos, radii, np.sum(radii)
    else:
        return None, None, -np.inf

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = -np.inf
    best_result = None
    
    # Initial guess: Grid
    pos_init = np.zeros((N, 2))
    idx = 0
    for i in range(5):
        for j in range(6):
            if idx >= N:
                break
            x = (i + 0.5) / 5.0
            y = (j + 0.5) / 6.0
            pos_init[idx] = [x, y]
            idx += 1
    
    # Try multiple random restarts
    rng = np.random.RandomState(42)
    n_restarts = 20  # Adjust based on time
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize positions
        if restart == 0:
            pos = pos_init.copy()
        else:
            pos = rng.uniform(0.1, 0.9, size=(N, 2))
            
        # Initialize radii to a small value
        radii = np.full(N, 0.05)
        
        # Combine into vector
        z0 = np.concatenate([pos.flatten(), radii])
        
        # Bounds: x, y in [0,1], r in [0, 0.5]
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.0, 0.5)) # r
            
        # Annealed penalty optimization
        best_z_for_restart = None
        best_obj_for_restart = np.inf
        
        # Start with high penalty, decrease to focus on objective
        lambda_start = 100.0
        lambda_end = 1000.0
        n_steps = 10
        
        for step in range(n_steps):
            if time.time() > deadline - 5:
                break
                
            # Linearly increase lambda to enforce constraints more strictly
            # Actually, usually we start with low lambda to move positions, then increase to fix overlaps.
            # Let's try increasing lambda.
            lambda_val = lambda_start + (lambda_end - lambda_start) * (step / n_steps)
            
            res = minimize(
                _objective, 
                z0, 
                args=(N, lambda_val), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            z0 = res.x
            obj_val = res.fun
            
            # Extract pos and radii
            pos_cur = z0[:2*N].reshape(N, 2)
            radii_cur = z0[2*N:]
            
            # Check validity and update best for this restart
            valid_pos, valid_radii, score = _fix_and_validate(pos_cur, radii_cur, N)
            if valid_pos is not None:
                if score > best_score:
                    best_score = score
                    best_result = np.zeros((N, 3))
                    best_result[:, :2] = valid_pos
                    best_result[:, 2] = valid_radii
                    
                if score > best_obj_for_restart: # Using score as proxy for objective quality
                    best_obj_for_restart = score
                    best_z_for_restart = z0.copy()
                    
            # If we have a very good solution, we might break early? 
            # No, let's let it run a bit more to polish.
            
    if best_result is None:
        # Fallback: Grid with small radii
        result = np.zeros((N, 3))
        idx = 0
        for i in range(5):
            for j in range(6):
                if idx >= N:
                    break
                x = (i + 0.5) / 5.0
                y = (j + 0.5) / 6.0
                r = 0.08
                result[idx, :] = [x, y, r]
                idx += 1
        return result
        
    return best_result

# EVOLVE-BLOCK-END
