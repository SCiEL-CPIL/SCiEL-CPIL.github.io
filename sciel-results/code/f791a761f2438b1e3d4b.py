"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _softplus(x, eps=1e-4):
    """Numerically stable softplus."""
    return np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))

def _objective_and_grad(params, n, soft_eps=1e-3):
    """
    params: array of shape (3n,) -> [x0,y0,r0, x1,y1,r1, ...]
    Returns (f, grad) for minimizing: -sum(r) + penalty * violations
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Objective: -sum(r)
    f = -np.sum(r)
    gx = np.zeros(n)
    gy = np.zeros(n)
    gr = -np.ones(n)
    
    # Boundary violations (smooth)
    # r_i <= x_i, r_i <= y_i, r_i <= 1-x_i, r_i <= 1-y_i
    bnd_viol = np.maximum(0, r - x) + np.maximum(0, r - y) + np.maximum(0, r - (1 - x)) + np.maximum(0, r - (1 - y))
    f += np.sum(bnd_viol)
    
    # Gradients for boundary (subgradient, use sign)
    gx += -np.sign(r - x) * (r > x) + np.sign(r - (1 - x)) * (r > (1 - x))
    gy += -np.sign(r - y) * (r > y) + np.sign(r - (1 - y)) * (r > (1 - y))
    gr += (r > x) + (r > y) + (r > (1 - x)) + (r > (1 - y))
    
    # Non-overlap violations
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
            min_dist = r[i] + r[j]
            viol = max(0, min_dist - dist)
            f += viol
            
            if viol > 0:
                # grad wrt x[i], y[i], r[i], x[j], y[j], r[j]
                # d/dx_i of (min_dist - dist) = -dx/dist
                inv_dist = 1.0 / dist
                gx[i] += -dx * inv_dist
                gy[i] += -dy * inv_dist
                gr[i] += 1.0
                gx[j] += dx * inv_dist
                gy[j] += dy * inv_dist
                gr[j] += 1.0
    
    grad = np.zeros(3 * n)
    grad[0::3] = gx
    grad[1::3] = gy
    grad[2::3] = gr
    
    return f, grad

def _check_validity(arr, n=N, tol=1e-7):
    """Check if circles are valid (inside square, non-overlapping)."""
    x = arr[:, 0]
    y = arr[:, 1]
    r = arr[:, 2]
    
    # Boundary
    if np.any(r < -tol) or np.any(x - r < -tol) or np.any(y - r < -tol) or np.any(1 - x - r < -tol) or np.any(1 - y - r < -tol):
        return False
    
    # Non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _penalize_and_fix(arr, n=N):
    """If invalid, shrink radii to make valid."""
    x = arr[:, 0].copy()
    y = arr[:, 1].copy()
    r = arr[:, 2].copy()
    
    # Ensure inside square
    for i in range(n):
        r[i] = min(r[i], x[i], y[i], 1 - x[i], 1 - y[i])
        r[i] = max(r[i], 0)
    
    # Fix overlaps by shrinking
    for _ in range(50):
        fixed = True
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                needed = (r[i] + r[j]) - dist
                if needed > 1e-10:
                    fixed = False
                    # Split the violation
                    total_r = r[i] + r[j]
                    if total_r > 0:
                        frac_i = r[i] / total_r
                        frac_j = r[j] / total_r
                        r[i] -= needed * frac_i
                        r[j] -= needed * frac_j
                    else:
                        r[i] = 0
                        r[j] = 0
        if fixed:
            break
    
    # Final boundary clamp
    for i in range(n):
        r[i] = min(r[i], x[i], y[i], 1 - x[i], 1 - y[i])
        r[i] = max(r[i], 0)
    
    return np.column_stack([x, y, r])

def _initial_hex(n):
    """Create a hexagonal-ish initial packing."""
    # Use a hexagonal grid pattern
    side = int(np.ceil(np.sqrt(n)))
    pts = []
    for row in range(side + 2):
        for col in range(side + 2):
            x = (col + 0.5) / (side + 1)
            y = (row + 0.5) / (side + 1)
            # Offset every other row
            if row % 2 == 1:
                x += 0.3 / (side + 1)
            if 0 <= x <= 1 and 0 <= y <= 1:
                r_init = 0.5 / (side + 1)
                pts.append((x, y, r_init))
    
    # Take first n
    pts = pts[:n]
    if len(pts) < n:
        # Fill remaining with random
        while len(pts) < n:
            pts.append((0.5, 0.5, 0.02))
    
    return np.array(pts)

def _random_init(n, rng):
    """Random initial configuration."""
    x = rng.uniform(0.1, 0.9, size=n)
    y = rng.uniform(0.1, 0.9, size=n)
    r = rng.uniform(0.01, 0.15, size=n)
    # Clamp to square
    r = np.minimum(r, np.minimum(x, y))
    r = np.minimum(r, np.minimum(1 - x, 1 - y))
    r = np.maximum(r, 0.001)
    return np.column_stack([x, y, r])

def _optimize_from_start(start_arr, n, max_iters=200, time_limit=10.0):
    """Run L-BFGS-B from a starting point."""
    t_start = time.time()
    params = start_arr.flatten()
    
    # Bounds: x in [0,1], y in [0,1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.0, 0.5))  # r
    
    def fun(p):
        if time.time() - t_start > time_limit:
            return 0.0
        f, _ = _objective_and_grad(p, n)
        return f
    
    def jac(p):
        if time.time() - t_start > time_limit:
            return np.zeros_like(p)
        _, g = _objective_and_grad(p, n)
        return g
    
    res = minimize(fun, params, jac=jac, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': max_iters, 'maxfun': max_iters * 10, 'ftol': 1e-10, 'gtol': 1e-8})
    
    if res.success or res.fun < 0:
        best_params = res.x
    else:
        best_params = params
    
    result = best_params.reshape(n, 3)
    return _penalize_and_fix(result, n)

def construct():
    t_global_start = time.time()
    time_budget = 140.0  # Leave 10s margin
    
    n = N
    rng = np.random.default_rng(42)
    
    best_score = -np.inf
    best_arr = None
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Hexagonal-ish
    inits.append(_initial_hex(n))
    
    # 2. Grid-based with jitter
    side = int(np.ceil(np.sqrt(n)))
    for i in range(side):
        for j in range(side):
            if len(inits) >= 5:
                break
    grid_pts = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95
            grid_pts.append((x, y, r))
    grid_pts = grid_pts[:n]
    if len(grid_pts) < n:
        while len(grid_pts) < n:
            grid_pts.append((0.5, 0.5, 0.05))
    inits.append(np.array(grid_pts))
    
    # 3-40: Random starts
    for _ in range(38):
        inits.append(_random_init(n, rng))
    
    # Optimize each start
    for idx, init in enumerate(inits):
        elapsed = time.time() - t_global_start
        if elapsed > time_budget - 5:
            break
        
        time_per = max(2.0, (time_budget - elapsed - 5) / max(1, len(inits) - idx))
        max_iters = max(50, int(100 * time_per / 10))
        
        result = _optimize_from_start(init, n, max_iters=max_iters, time_limit=time_per)
        
        score = np.sum(result[:, 2])
        
        if score > best_score:
            best_score = score
            best_arr = result
    
    # Final polish on the best
    if best_arr is not None:
        elapsed = time.time() - t_global_start
        remaining = time_budget - elapsed
        if remaining > 5:
            polished = _optimize_from_start(best_arr, n, max_iters=300, time_limit=remaining - 2)
            polished_score = np.sum(polished[:, 2])
            if polished_score > best_score:
                best_arr = polished
                best_score = polished_score
    
    if best_arr is None:
        # Fallback
        best_arr = _initial_hex(n)
    
    # Final validity check and fix
    best_arr = _penalize_and_fix(best_arr, n)
    
    # Ensure it's a valid numpy array of shape (n, 3)
    best_arr = np.asarray(best_arr, dtype=np.float64)
    if best_arr.shape != (n, 3):
        best_arr = _initial_hex(n)
    
    return best_arr

# EVOLVE-BLOCK-END
