"""Circle packing in perimeter-4 rectangle - multi-start local search with force-directed updates."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _check_valid(circles, tol=1e-9):
    """Check if circles are valid: disjoint interiors, bounding rect perimeter <= 4."""
    if circles.shape[0] != N:
        return False
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs < -tol):
        return False
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    w = max_x - min_x
    h = max_y - min_y
    if 2 * (w + h) > 4 + tol:
        return False
    # Check pairwise separation
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            if d < rs[i] + rs[j] - tol:
                return False
    return True

def _penalty_objective(params, lam=1e4):
    """
    Objective: minimize negative sum of radii + penalties for constraint violations.
    params: flattened array of shape (3N,)
    """
    n = N
    x = params.reshape(n, 3)
    xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
    
    # Radii positivity
    neg_r = np.maximum(0, -rs)
    penalty_r = np.sum(neg_r ** 2)
    
    # Bounding box perimeter
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    w = max_x - min_x
    h = max_y - min_y
    perim = 2 * (w + h)
    perim_viol = max(0, perim - 4)
    penalty_perim = perim_viol ** 2
    
    # Separation constraints - vectorized
    penalty_sep = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            req = rs[i] + rs[j]
            viol = max(0, req - d)
            penalty_sep += viol ** 2
    
    total_penalty = lam * (penalty_r + penalty_perim + penalty_sep)
    return -np.sum(rs) + total_penalty

def _force_directed_relax(circles, n_steps=200, step_size=0.01, deadline=None):
    """
    Force-directed relaxation: push overlapping circles apart and scale to fit perimeter.
    Returns updated circles.
    """
    circles = circles.copy()
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute forces from overlaps
        forces_x = np.zeros(N)
        forces_y = np.zeros(N)
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[j] - xs[i]
                dy = ys[j] - ys[i]
                d = np.sqrt(dx**2 + dy**2)
                req = rs[i] + rs[j]
                if d < req and d > 1e-12:
                    # Push apart
                    overlap = req - d
                    fx = overlap * dx / d
                    fy = overlap * dy / d
                    forces_x[i] -= fx
                    forces_y[i] -= fy
                    forces_x[j] += fx
                    forces_y[j] += fy
                elif d <= 1e-12:
                    # Random push if exactly overlapping
                    angle = np.random.uniform(0, 2 * np.pi)
                    fx = overlap * np.cos(angle) if d < req else 0
                    fy = overlap * np.sin(angle) if d < req else 0
                    forces_x[i] -= fx
                    forces_y[i] -= fy
                    forces_x[j] += fx
                    forces_y[j] += fy
        
        # Apply forces
        xs += step_size * forces_x
        ys += step_size * forces_y
        
        # Scale to fit perimeter constraint
        min_x = np.min(xs - rs)
        max_x = np.max(xs + rs)
        min_y = np.min(ys - rs)
        max_y = np.max(ys + rs)
        w = max_x - min_x
        h = max_y - min_y
        perim = 2 * (w + h)
        
        if perim > 4:
            scale = 2.0 / perim
            # Scale positions relative to center
            cx = (min_x + max_x) / 2
            cy = (min_y + max_y) / 2
            xs = cx + (xs - cx) * scale
            ys = cy + (ys - cy) * scale
            rs = rs * scale
        
        circles[:, 0] = xs
        circles[:, 1] = ys
        circles[:, 2] = rs
    
    return circles

def _optimize_from_init(circles0, deadline, rng, max_lam_stages=4):
    """
    Run optimization from an initial configuration.
    Uses penalty method with increasing lambda, then force-directed polish.
    """
    params0 = circles0.flatten()
    best_circles = circles0.copy()
    best_sum = np.sum(circles0[:, 2])
    
    # Multi-stage penalty optimization
    lam_stages = [1e3, 1e4, 1e5, 1e6]
    
    for lam in lam_stages:
        if time.time() > deadline:
            break
        
        def obj(p):
            return _penalty_objective(p, lam=lam)
        
        try:
            res = minimize(obj, params0, method='L-BFGS-B', 
                          options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-10})
            params0 = res.x
            circles = res.x.reshape(N, 3)
            
            if _check_valid(circles):
                s = np.sum(circles[:, 2])
                if s > best_sum:
                    best_sum = s
                    best_circles = circles.copy()
        except Exception:
            continue
    
    # Force-directed polish
    if time.time() < deadline - 5:
        polished = _force_directed_relax(best_circles, n_steps=300, step_size=0.005, deadline=deadline)
        if _check_valid(polished):
            s = np.sum(polished[:, 2])
            if s > best_sum:
                best_sum = s
                best_circles = polished.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 10:
        def obj_final(p):
            return _penalty_objective(p, lam=1e7)
        
        try:
            res = minimize(obj_final, best_circles.flatten(), method='L-BFGS-B',
                          options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12})
            circles = res.x.reshape(N, 3)
            if _check_valid(circles):
                s = np.sum(circles[:, 2])
                if s > best_sum:
                    best_sum = s
                    best_circles = circles.copy()
        except Exception:
            pass
    
    return best_circles, best_sum

def _make_grid_init(nx, ny, aspect, rng):
    """Create a grid-based initial configuration."""
    w = 2.0 / (1 + aspect)
    h = 2.0 - w
    
    circles = np.zeros((N, 3))
    count = 0
    for i in range(nx):
        for j in range(ny):
            if count >= N:
                break
            cx = (i + 0.5) * w / nx
            cy = (j + 0.5) * h / ny
            r = min(w / nx, h / ny) * 0.45
            # Small random perturbation
            cx += rng.uniform(-0.01, 0.01)
            cy += rng.uniform(-0.01, 0.01)
            r *= (1 + rng.uniform(-0.05, 0.05))
            circles[count] = [cx, cy, r]
            count += 1
    
    # If we have fewer than N, fill with small circles
    while count < N:
        cx = rng.uniform(0, w)
        cy = rng.uniform(0, h)
        r = 0.02
        circles[count] = [cx, cy, r]
        count += 1
    
    return circles

def _make_hexagonal_init(nx, ny, aspect, rng):
    """Create a hexagonal-lattice-based initial configuration."""
    w = 2.0 / (1 + aspect)
    h = 2.0 - w
    
    circles = np.zeros((N, 3))
    count = 0
    
    # Hexagonal spacing
    dx = w / nx
    dy = dx * np.sqrt(3) / 2
    r = dx * 0.45
    
    for i in range(nx):
        for j in range(ny):
            if count >= N:
                break
            offset = (dx / 2) if (j % 2 == 1) else 0
            cx = (i + 0.5) * dx + offset
            cy = (j + 0.5) * dy
            # Keep within bounds
            if cx + r > w or cy + r > h:
                continue
            cx += rng.uniform(-0.01, 0.01)
            cy += rng.uniform(-0.01, 0.01)
            r *= (1 + rng.uniform(-0.05, 0.05))
            circles[count] = [cx, cy, r]
            count += 1
    
    # Fill remaining with small circles
    while count < N:
        cx = rng.uniform(0, w)
        cy = rng.uniform(0, h)
        r = 0.02
        circles[count] = [cx, cy, r]
        count += 1
    
    return circles

def _make_random_init(aspect, rng):
    """Create a random initial configuration."""
    w = 2.0 / (1 + aspect)
    h = 2.0 - w
    
    circles = np.zeros((N, 3))
    for i in range(N):
        cx = rng.uniform(0.05, w - 0.05)
        cy = rng.uniform(0.05, h - 0.05)
        r = rng.uniform(0.03, min(w, h) * 0.2)
        circles[i] = [cx, cy, r]
    
    return circles

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    rng = np.random.RandomState(42)
    
    best_circles = None
    best_sum = 0.0
    
    # Try multiple aspect ratios and initialization strategies
    aspect_ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0]
    
    # Grid configurations
    grid_configs = [(4, 5), (5, 4), (3, 7), (7, 3), (6, 4), (4, 6), (5, 5)]
    
    attempts = 0
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        # Try grid init
        for nx, ny in grid_configs[:4]:
            if time.time() > deadline:
                break
            attempts += 1
            init_circles = _make_grid_init(nx, ny, ar, rng)
            circles, s = _optimize_from_init(init_circles, deadline, rng)
            if s > best_sum:
                best_sum = s
                best_circles = circles.copy()
        
        # Try hexagonal init
        if time.time() < deadline - 10:
            for nx, ny in [(4, 5), (5, 4), (6, 4)]:
                if time.time() > deadline:
                    break
                attempts += 1
                init_circles = _make_hexagonal_init(nx, ny, ar, rng)
                circles, s = _optimize_from_init(init_circles, deadline, rng)
                if s > best_sum:
                    best_sum = s
                    best_circles = circles.copy()
        
        # Try random init
        if time.time() < deadline - 10:
            attempts += 1
            init_circles = _make_random_init(ar, rng)
            circles, s = _optimize_from_init(init_circles, deadline, rng)
            if s > best_sum:
                best_sum = s
                best_circles = circles.copy()
    
    if best_circles is not None:
        return best_circles
    
    # Fallback: simple grid
    w, h = 1.0, 1.0
    nx, ny = 5, 5
    cells = []
    for i in range(nx):
        for j in range(ny):
            if len(cells) >= N:
                break
            cells.append([(i + 0.5) * w / nx, (j + 0.5) * h / ny, min(w/nx, h/ny) * 0.4])
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
