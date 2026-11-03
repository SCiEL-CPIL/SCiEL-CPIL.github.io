"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _objective(params, W, H):
    """Objective: maximize sum of radii subject to constraints.
    We minimize negative sum of radii.
    params: flat array [x0,y0,r0, x1,y1,r1, ...]
    Returns: -sum(r) + penalty for violations
    """
    n = N
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Penalty for circles outside rectangle
    penalty = 0.0
    for i in range(n):
        if x[i] < r[i]:
            penalty += (r[i] - x[i]) ** 2 * 100
        if x[i] > W - r[i]:
            penalty += (x[i] - (W - r[i])) ** 2 * 100
        if y[i] < r[i]:
            penalty += (r[i] - y[i]) ** 2 * 100
        if y[i] > H - r[i]:
            penalty += (y[i] - (H - r[i])) ** 2 * 100
        if r[i] < 0:
            penalty += r[i] ** 2 * 100
    
    # Penalty for overlapping circles
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2 * 100
    
    return -np.sum(r) + penalty

def _construct_for_aspect(W, H, max_iter=2000, seed=None):
    """Try to pack N circles in rectangle W x H (perimeter 4)."""
    if seed is None:
        seed = 42
    rng = np.random.RandomState(seed)
    
    best_params = None
    best_val = np.inf
    
    # Multiple random restarts
    n_restarts = 10
    for restart in range(n_restarts):
        # Initial guess: random positions and small radii
        x0 = rng.uniform(0.1, W - 0.1, N)
        y0 = rng.uniform(0.1, H - 0.1, N)
        r0 = np.full(N, min(W, H) / (2 * N) * 0.8)
        
        params0 = np.zeros(3 * N)
        params0[0::3] = x0
        params0[1::3] = y0
        params0[2::3] = r0
        
        res = minimize(_objective, params0, args=(W, H), method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    
    if best_params is None:
        return None, -np.inf
    
    x = best_params[0::3]
    y = best_params[1::3]
    r = best_params[2::3]
    
    # Check validity
    valid = True
    for i in range(N):
        if x[i] < -1e-6 or x[i] > W + 1e-6 or y[i] < -1e-6 or y[i] > H + 1e-6:
            valid = False
            break
        if r[i] < -1e-6:
            valid = False
            break
    if valid:
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - 1e-6:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        return None, -np.inf
    
    sum_r = np.sum(r)
    return np.column_stack([x, y, r]), sum_r

def construct():
    """Pack 21 circles in a perimeter-4 rectangle, maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    best_result = None
    best_score = -np.inf
    
    # Search over aspect ratios
    # Perimeter = 2(W + H) = 4 => W + H = 2
    # Let aspect ratio a = W/H, so W = a*H, H = 2/(1+a), W = 2a/(1+a)
    aspect_ratios = np.linspace(0.5, 2.0, 15)
    
    for a in aspect_ratios:
        if time.time() > deadline:
            break
        
        H = 2.0 / (1.0 + a)
        W = a * H
        
        # Try multiple seeds for robustness
        for seed in [42, 123, 456]:
            if time.time() > deadline:
                break
            
            result, score = _construct_for_aspect(W, H, max_iter=1500, seed=seed)
            if result is not None and score > best_score:
                best_score = score
                best_result = result
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
