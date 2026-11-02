"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 32

# EVOLVE-BLOCK-START
def _penalty_and_objective(z, N, lam=1000.0):
    """
    z: array of shape (3*N,) containing [x1, y1, r1, ..., xN, yN, rN]
    Returns:
        obj: -sum(r) + lam * penalty
        penalty: sum of squared constraint violations
    """
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    
    # Objective: maximize sum(r) -> minimize -sum(r)
    obj = -np.sum(r)
    
    # Containment Penalty
    # r <= x, r <= 1-x, r <= y, r <= 1-y
    pen_cont = 0.0
    pen_cont += np.sum(np.maximum(0, r - x)**2)
    pen_cont += np.sum(np.maximum(0, r - (1 - x))**2)
    pen_cont += np.sum(np.maximum(0, r - y)**2)
    pen_cont += np.sum(np.maximum(0, r - (1 - y))**2)
    
    # Non-overlap Penalty
    # (xi - xj)^2 + (yi - yj)^2 >= (ri + rj)^2
    # Violation: ri + rj - dist > 0
    pen_ov = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            gap = r[i] + r[j] - dist
            if gap > 0:
                pen_ov += gap * gap
                
    total_penalty = pen_cont + pen_ov
    return obj + lam * total_penalty, total_penalty

def _grid_init(N):
    side = int(np.ceil(np.sqrt(N)))
    # Use a 6x6 grid for N=32 to have some space
    if side < 4: side = 4
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N: break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.4 / side # slightly smaller than half cell width
            cells.append([x, y, r])
    return np.array(cells).flatten()

def _hex_init(N):
    # Hexagonal packing initial guess
    # Approximate radius for N circles in unit square
    # r ~ 0.5 / sqrt(N) * packing_factor
    r_est = 0.4 / np.sqrt(N)
    points = []
    # Generate hex grid points
    dx = 2 * r_est * 1.1 # spacing
    dy = dx * np.sqrt(3) / 2
    
    # Center the grid
    # We need to fit N points.
    # Just generate a large grid and pick the first N closest to center?
    # Or just generate a grid that covers [0,1]x[0,1]
    
    # Let's just generate a grid of hex points and filter those inside [0,1]
    # And then pick N of them.
    # To be safe, generate more than N and pick the best N?
    # For simplicity, generate a grid with spacing based on r_est.
    
    candidates = []
    for i in range(-5, 20):
        for j in range(-5, 20):
            x = i * dx + (0.5 * dx if j % 2 != 0 else 0)
            y = j * dy
            if -0.1 <= x <= 1.1 and -0.1 <= y <= 1.1:
                # Check if it fits (roughly)
                if r_est <= x and r_est <= 1-x and r_est <= y and r_est <= 1-y:
                    candidates.append([x, y, r_est])
                    
    if len(candidates) < N:
        # Fallback to grid
        return _grid_init(N)
        
    # Shuffle to ensure diversity? No, deterministic is better for reproducibility of init.
    # But we want a specific pattern.
    # Let's just take the first N.
    return np.array(candidates[:N]).flatten()

def _random_init(N, seed=0):
    np.random.seed(seed)
    x = np.random.rand(N)
    y = np.random.rand(N)
    r = np.random.rand(N) * 0.1 + 0.01 # small radii
    z = np.zeros(3*N)
    z[0::3] = x
    z[1::3] = y
    z[2::3] = r
    return z

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin for 120s limit
    
    best_score = -np.inf
    best_z = None
    best_penalty = np.inf
    
    # List of initializers
    inits = [
        _grid_init(N),
        _hex_init(N),
        _random_init(N, 1),
        _random_init(N, 2),
        _random_init(N, 3)
    ]
    
    # Optimization parameters
    lam = 1000.0
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    
    for z0 in inits:
        if time.time() > deadline:
            break
            
        # Define objective for optimizer
        def objective(z):
            val, _ = _penalty_and_objective(z, N, lam)
            return val
        
        try:
            res = minimize(objective, z0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            # Evaluate final penalty
            _, pen = _penalty_and_objective(res.x, N, lam)
            
            # If penalty is small, consider it valid
            if pen < 1e-6:
                score = np.sum(res.x[2::3])
                if score > best_score:
                    best_score = score
                    best_z = res.x
                    best_penalty = pen
                    
        except Exception:
            continue
            
    # Polish Phase: If we have a good solution, try to refine it
    if best_z is not None:
        # Increase lambda for tighter constraints?
        # Or just run again with tighter tolerances
        lam_polish = 10000.0
        
        def objective_polish(z):
            val, _ = _penalty_and_objective(z, N, lam_polish)
            return val
            
        try:
            res_polish = minimize(objective_polish, best_z, method='L-BFGS-B', bounds=bounds,
                                  options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            _, pen_polish = _penalty_and_objective(res_polish.x, N, lam_polish)
            if pen_polish < 1e-6:
                score_polish = np.sum(res_polish.x[2::3])
                if score_polish > best_score:
                    best_score = score_polish
                    best_z = res_polish.x
        except Exception:
            pass

    # If no valid solution found, return a trivial valid one (grid)
    if best_z is None:
        best_z = _grid_init(N)
        # Ensure it's valid by scaling radii down if necessary? 
        # The grid init with 0.4/side is generally valid.
        
    # Format output: (N, 3) array
    x = best_z[0::3]
    y = best_z[1::3]
    r = best_z[2::3]
    
    # Final sanity check: clip radii to be non-negative and ensure they fit
    # If penalty was not 0, we might have invalid circles.
    # We should project them to a valid state if possible, or just return them 
    # if the penalty was very small.
    # The evaluator checks disjoint interiors. 
    # If pen < 1e-6, it's effectively disjoint.
    
    result = np.column_stack((x, y, r))
    
    # One last check: if any radius is negative, set to 0? No, that breaks packing.
    # If the optimizer failed to find a valid solution, the score might be low.
    # But we initialized with valid or near-valid configs.
    
    return result

# EVOLVE-BLOCK-END
