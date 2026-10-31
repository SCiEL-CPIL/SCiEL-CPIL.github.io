"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 21

# EVOLVE-BLOCK-START

def _initial_grid(n, W, H, n_cols=None):
    """Generate initial circle positions in a grid pattern within [0,W]x[0,H]."""
    if n_cols is None:
        n_cols = int(math.ceil(math.sqrt(n * W / H)))
    n_rows = int(math.ceil(n / n_cols))
    dx = W / n_cols
    dy = H / n_rows
    cells = []
    for i in range(n):
        c = i % n_cols
        r = i // n_cols
        x = (c + 0.5) * dx
        y = (r + 0.5) * dy
        r0 = 0.5 * min(dx, dy) - 0.005
        cells.append([x, y, r0])
    return np.array(cells[:n])


def _objective_and_grads(x, W, H, n):
    """
    x: [2n + 2] vector: [x_0..x_{n-1}, y_0..y_{n-1}, W, H]
    Returns (objective, grad) where objective = -sum(r_i)
    """
    # Unpack
    cx = x[:n]
    cy = x[n:2*n]
    w = x[2*n]
    h = x[2*n+1]
    
    # Radii: min distance to walls and to other circles
    radii = np.full(n, np.inf)
    
    # Wall constraints
    radii = np.minimum(radii, cx)       # left
    radii = np.minimum(radii, w - cx)   # right
    radii = np.minimum(radii, cy)       # bottom
    radii = np.minimum(radii, h - cy)   # top
    
    # Circle-circle constraints
    for i in range(n):
        for j in range(i+1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = math.sqrt(dx*dx + dy*dy)
            radii[i] = min(radii[i], dist / 2.0)
            radii[j] = min(radii[j], dist / 2.0)
    
    obj = -np.sum(radii)
    
    # Compute gradient via finite differences (cheap for n=21, ~44 variables)
    grad = np.zeros_like(x)
    eps = 1e-7
    for k in range(len(x)):
        xp = x.copy()
        xp[k] += eps
        obj_p, _ = _objective_only(xp, n)
        grad[k] = (obj_p - obj) / eps
    
    return obj, grad


def _objective_only(x, n):
    """Compute objective value only (no gradient)."""
    cx = x[:n]
    cy = x[n:2*n]
    w = x[2*n]
    h = x[2*n+1]
    
    radii = np.full(n, np.inf)
    radii = np.minimum(radii, cx)
    radii = np.minimum(radii, w - cx)
    radii = np.minimum(radii, cy)
    radii = np.minimum(radii, h - cy)
    
    for i in range(n):
        for j in range(i+1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = math.sqrt(dx*dx + dy*dy)
            radii[i] = min(radii[i], dist / 2.0)
            radii[j] = min(radii[j], dist / 2.0)
    
    return -np.sum(radii)


def _optimize_fixed_aspect(W, H, n, time_limit=10):
    """Optimize circle positions and radii for fixed rectangle W x H."""
    from scipy.optimize import minimize
    
    # Initial guess: grid
    x0 = _initial_grid(n, W, H)
    x0_flat = np.concatenate([x0[:, 0], x0[:, 1]])
    
    # Bounds
    bounds = []
    for _ in range(n):
        bounds.append((0.0, W))
    for _ in range(n):
        bounds.append((0.0, H))
    
    # Objective: maximize sum of radii = minimize -sum(radii)
    # Radii determined by min distance to walls and other circles
    def obj(x):
        cx = x[:n]
        cy = x[n:2*n]
        radii = np.full(n, np.inf)
        radii = np.minimum(radii, cx)
        radii = np.minimum(radii, W - cx)
        radii = np.minimum(radii, cy)
        radii = np.minimum(radii, H - cy)
        for i in range(n):
            for j in range(i+1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = math.sqrt(dx*dx + dy*dy)
                radii[i] = min(radii[i], dist / 2.0)
                radii[j] = min(radii[j], dist / 2.0)
        return -np.sum(radii)
    
    result = minimize(obj, x0_flat, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
    
    best_x = result.x
    best_val = -result.fun
    return best_x, best_val


def _optimize_free_aspect(n, time_limit=15):
    """
    Optimize both circle positions AND rectangle dimensions (W, H) 
    subject to 2(W+H) <= 4, i.e., W+H <= 2.
    """
    from scipy.optimize import minimize
    
    # Parameterize: W in (0.2, 1.8), H = 2 - W
    # x = [x_0..x_{n-1}, y_0..y_{n-1}, W]
    # H = 2 - W
    
    def obj(x):
        cx = x[:n]
        cy = x[n:2*n]
        w = x[2*n]
        h = 2.0 - w
        if w <= 0.01 or h <= 0.01:
            return 1e10
        radii = np.full(n, np.inf)
        radii = np.minimum(radii, cx)
        radii = np.minimum(radii, w - cx)
        radii = np.minimum(radii, cy)
        radii = np.minimum(radii, h - cy)
        for i in range(n):
            for j in range(i+1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = math.sqrt(dx*dx + dy*dy)
                radii[i] = min(radii[i], dist / 2.0)
                radii[j] = min(radii[j], dist / 2.0)
        return -np.sum(radii)
    
    # Try multiple initial aspects
    best_x = None
    best_val = -1e10
    
    aspects = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6]
    for aspect in aspects:
        # W = aspect / (1 + aspect) * 2, H = 1/(1+aspect)*2
        W0 = 2.0 * aspect / (1.0 + aspect)
        H0 = 2.0 / (1.0 + aspect)
        
        x0_grid = _initial_grid(n, W0, H0)
        x0 = np.concatenate([x0_grid[:, 0], x0_grid[:, 1], [W0]])
        
        bounds = []
        for _ in range(n):
            bounds.append((0.0, 1.8))
        for _ in range(n):
            bounds.append((0.0, 1.8))
        bounds.append((0.3, 1.7))  # W range
        
        try:
            result = minimize(obj, x0, method='Nelder-Mead',
                            options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9})
            val = -result.fun
            if val > best_val:
                best_val = val
                best_x = result.x
        except Exception:
            pass
    
    if best_x is None:
        # Fallback
        x0 = np.concatenate([_initial_grid(n, 1.0, 1.0)[:, 0], 
                           _initial_grid(n, 1.0, 1.0)[:, 1], [1.0]])
        return x0, _objective_only(x0, n)
    
    return best_x, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    n = N
    best_x = None
    best_val = -1e10
    
    # Phase 1: Free aspect ratio optimization
    try:
        x, val = _optimize_free_aspect(n, time_limit=60)
        if val > best_val:
            best_val = val
            best_x = x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        # Not enough time for more phases
        pass
    
    # Phase 2: Try to improve with fixed-aspect optimizations at various aspects
    if time.time() < deadline - 30:
        for aspect in [1.0, 0.9, 1.1, 0.8, 1.25]:
            if time.time() > deadline - 15:
                break
            W = 2.0 * aspect / (1.0 + aspect)
            H = 2.0 / (1.0 + aspect)
            try:
                x_fixed, val_fixed = _optimize_fixed_aspect(W, H, n, time_limit=8)
                # Convert back to free-aspect representation
                x_full = np.concatenate([x_fixed, [W]])
                val_check = -_objective_only(x_full, n)
                if val_check > best_val:
                    best_val = val_check
                    best_x = x_full
            except Exception:
                pass
    
    # Phase 3: Local refinement with different initializations
    if time.time() < deadline - 15 and best_x is not None:
        from scipy.optimize import minimize
        
        def obj_free(x):
            cx = x[:n]
            cy = x[n:2*n]
            w = x[2*n]
            h = 2.0 - w
            if w <= 0.01 or h <= 0.01:
                return 1e10
            radii = np.full(n, np.inf)
            radii = np.minimum(radii, cx)
            radii = np.minimum(radii, w - cx)
            radii = np.minimum(radii, cy)
            radii = np.minimum(radii, h - cy)
            for i in range(n):
                for j in range(i+1, n):
                    dx = cx[i] - cx[j]
                    dy = cy[i] - cy[j]
                    dist = math.sqrt(dx*dx + dy*dy)
                    radii[i] = min(radii[i], dist / 2.0)
                    radii[j] = min(radii[j], dist / 2.0)
            return -np.sum(radii)
        
        # Perturb best solution and re-optimize
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            np.random.seed(42 + trial)
            x_pert = best_x.copy()
            x_pert[:2*n] += np.random.randn(2*n) * 0.02
            x_pert[2*n] = np.clip(x_pert[2*n], 0.3, 1.7)
            
            try:
                result = minimize(obj_free, x_pert, method='Nelder-Mead',
                                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
                val = -result.fun
                if val > best_val:
                    best_val = val
                    best_x = result.x
            except Exception:
                pass
    
    # Build output: array of (x, y, radius) for each circle
    if best_x is None:
        # Fallback to simple grid
        cells = _initial_grid(n, 1.0, 1.0)
        return cells
    
    cx = best_x[:n]
    cy = best_x[n:2*n]
    w = best_x[2*n]
    h = 2.0 - w
    
    # Compute final radii
    radii = np.full(n, np.inf)
    radii = np.minimum(radii, cx)
    radii = np.minimum(radii, w - cx)
    radii = np.minimum(radii, cy)
    radii = np.minimum(radii, h - cy)
    for i in range(n):
        for j in range(i+1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = math.sqrt(dx*dx + dy*dy)
            radii[i] = min(radii[i], dist / 2.0)
            radii[j] = min(radii[j], dist / 2.0)
    
    # Ensure radii are non-negative and small enough
    radii = np.maximum(radii, 0.0)
    
    # Clamp positions to be within [r_i, w - r_i] x [r_i, h - r_i]
    for i in range(n):
        r = radii[i]
        cx[i] = np.clip(cx[i], r, w - r)
        cy[i] = np.clip(cy[i], r, h - r)
    
    # Recompute radii after clamping
    radii = np.full(n, np.inf)
    radii = np.minimum(radii, cx)
    radii = np.minimum(radii, w - cx)
    radii = np.minimum(radii, cy)
    radii = np.minimum(radii, h - cy)
    for i in range(n):
        for j in range(i+1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = math.sqrt(dx*dx + dy*dy)
            radii[i] = min(radii[i], dist / 2.0)
            radii[j] = min(radii[j], dist / 2.0)
    radii = np.maximum(radii, 0.0)
    
    result = np.column_stack([cx, cy, radii])
    return result

# EVOLVE-BLOCK-END
