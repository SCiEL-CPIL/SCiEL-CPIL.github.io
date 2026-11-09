"""Rewritten program: local-search circle packing in the unit square."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 26
DEADLINE = 115.0  # safety margin before the 120 s kill


def _max_radius(x, y, other_idx, xs, ys, rs):
    """Largest radius at (x, y) given other circles and square walls."""
    r = min(x, 1.0 - x, y, 1.0 - y)
    for j in other_idx:
        d = math.hypot(x - xs[j], y - ys[j])
        r = min(r, d - rs[j])
    return max(r, 1e-9)


def _score(xs, ys, rs):
    return float(np.sum(rs))


def _validate(xs, ys, rs, tol=1e-7):
    if len(xs) != N:
        return False
    for i in range(N):
        if xs[i] < -tol or xs[i] > 1.0 + tol or ys[i] < -tol or ys[i] > 1.0 + tol:
            return False
        if rs[i] < 0:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            d = math.hypot(xs[i] - xs[j], ys[i] - ys[j])
            if d < rs[i] + rs[j] - tol:
                return False
    return True


def _clamp(x, y, r):
    x = min(max(x, r), 1.0 - r)
    y = min(max(y, r), 1.0 - r)
    return x, y


def _local_search(xs, ys, rs, deadline):
    """Iterative improvement: move each circle to maximize its own radius."""
    best_score = _score(xs, ys, rs)
    n = len(xs)
    # Precompute neighbor indices for efficiency
    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            old_r = rs[i]
            best_r = old_r
            best_x, best_y = xs[i], ys[i]
            other = [j for j in range(n) if j != i]
            # Random perturbations with adaptive step size
            step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002]
            for step in step_sizes:
                for _ in range(10):
                    dx = np.random.uniform(-step, step)
                    dy = np.random.uniform(-step, step)
                    nx, ny = xs[i] + dx, ys[i] + dy
                    nx, ny = _clamp(nx, ny, old_r)
                    new_r = _max_radius(nx, ny, other, xs, ys, rs)
                    if new_r > best_r + 1e-12:
                        best_r = new_r
                        best_x, best_y = nx, ny
            if best_r > old_r + 1e-12:
                xs[i], ys[i], rs[i] = best_x, best_y, best_r
                improved = True
                new_score = _score(xs, ys, rs)
                if new_score > best_score:
                    best_score = new_score
        if not improved:
            # Try larger random jumps to escape local optima
            for _ in range(30):
                if time.time() >= deadline:
                    break
                i = np.random.randint(n)
                old_x, old_y, old_r = xs[i], ys[i], rs[i]
                nx = np.random.uniform(0.05, 0.95)
                ny = np.random.uniform(0.05, 0.95)
                nx, ny = _clamp(nx, ny, old_r)
                other = [j for j in range(n) if j != i]
                new_r = _max_radius(nx, ny, other, xs, ys, rs)
                if new_r > old_r + 1e-12:
                    xs[i], ys[i], rs[i] = nx, ny, new_r
                    improved = True
                    new_score = _score(xs, ys, rs)
                    if new_score > best_score:
                        best_score = new_score
            if not improved:
                break
    return xs, ys, rs, best_score


def _initial_grid():
    """Create an initial grid-based packing."""
    side = int(math.ceil(math.sqrt(N)))
    xs, ys, rs = [], [], []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            xs.append(x)
            ys.append(y)
            rs.append(r)
            count += 1
    return np.array(xs), np.array(ys), np.array(rs)


def _initial_hexagonal():
    """Create a hexagonal lattice initial packing for denser starts."""
    xs, ys, rs = [], [], []
    # Hexagonal packing: rows offset, spacing optimized for density
    # For 26 circles, use a hexagonal arrangement
    # Try different row/column configurations
    configs = [
        (5, 5), (6, 5), (5, 6), (7, 4), (4, 7), (6, 4), (4, 6), (8, 4), (4, 8)
    ]
    
    best_config = None
    best_min_radius = 0
    
    for nrows, ncols in configs:
        if nrows * ncols < N:
            continue
        # Hexagonal spacing
        dx = 1.0 / (ncols + 0.5)  # horizontal spacing
        dy = dx * math.sqrt(3) / 2  # vertical spacing for hexagonal
        
        # Adjust to fit in unit square
        total_width = (ncols - 1) * dx + dx  # center to center
        total_height = (nrows - 1) * dy + dy
        
        # Scale to fit
        scale_x = 1.0 / total_width
        scale_y = 1.0 / total_height
        scale = min(scale_x, scale_y)
        
        dx *= scale
        dy *= scale
        
        # Center the pattern
        offset_x = (1.0 - (ncols - 1) * dx) / 2
        offset_y = (1.0 - (nrows - 1) * dy) / 2
        
        cx, cy, cr = [], [], []
        count = 0
        for i in range(nrows):
            for j in range(ncols):
                if count >= N:
                    break
                x = offset_x + j * dx
                y = offset_y + i * dy
                # Offset every other row
                if i % 2 == 1:
                    x += dx / 2
                
                # Max radius at this position
                r = min(x, 1.0 - x, y, 1.0 - y)
                for k in range(len(cx)):
                    d = math.hypot(x - cx[k], y - cy[k])
                    r = min(r, d - cr[k])
                
                r = max(r, 0.01)
                cx.append(x)
                cy.append(y)
                cr.append(r)
                count += 1
        
        min_r = min(cr) if cr else 0
        if min_r > best_min_radius:
            best_min_radius = min_r
            best_config = (cx, cy, cr)
    
    if best_config is not None:
        return np.array(best_config[0]), np.array(best_config[1]), np.array(best_config[2])
    
    # Fallback to grid
    return _initial_grid()


def _initial_random():
    """Create a random initial packing with small circles."""
    xs, ys, rs = [], [], []
    for _ in range(N):
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        r = 0.02
        xs.append(x)
        ys.append(y)
        rs.append(r)
    return np.array(xs), np.array(ys), np.array(rs)


def _polish_smooth(xs, ys, rs, deadline, temp=0.01):
    """Polish using smooth surrogate for the max radius objective."""
    n = len(xs)
    best_score = _score(xs, ys, rs)
    
    # Use L-BFGS-B to optimize positions and radii jointly
    # Variable vector: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = xs[i]
        x0[3*i+1] = ys[i]
        x0[3*i+2] = rs[i]
    
    def objective(v):
        if time.time() >= deadline:
            return 1e10
        # Unpack
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        
        # Penalty for violations
        penalty = 0.0
        # Wall constraints
        for i in range(n):
            if x[i] < r[i] or x[i] > 1.0 - r[i]:
                penalty += 100 * (min(x[i] - r[i], r[i] - x[i] + 1.0, 1e-6))
            if y[i] < r[i] or y[i] > 1.0 - r[i]:
                penalty += 100 * (min(y[i] - r[i], r[i] - y[i] + 1.0, 1e-6))
        
        # Overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                d = math.hypot(x[i] - x[j], y[i] - y[j])
                if d < r[i] + r[j]:
                    penalty += 100 * (r[i] + r[j] - d)
        
        # Negative sum of radii (we minimize)
        return -np.sum(r) + penalty
    
    # Bounds: r > 0, x,y in [0,1]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((1e-6, 0.5))  # r
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 100, 'maxfun': 10000, 'ftol': 1e-12})
        if result.success and result.fun < -best_score + 1e-9:
            # Extract improved solution
            new_xs = result.x[0::3]
            new_ys = result.x[1::3]
            new_rs = result.x[2::3]
            
            # Validate and clamp
            valid = True
            for i in range(n):
                new_xs[i] = min(max(new_xs[i], new_rs[i]), 1.0 - new_rs[i])
                new_ys[i] = min(max(new_ys[i], new_rs[i]), 1.0 - new_rs[i])
            
            if _validate(new_xs, new_ys, new_rs):
                new_score = _score(new_xs, new_ys, new_rs)
                if new_score > best_score:
                    return new_xs, new_ys, new_rs, new_score
    except Exception:
        pass
    
    return xs, ys, rs, best_score


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Try multiple restarts with different initializations
    best_xs, best_ys, best_rs = None, None, None
    best_score = -1.0

    # Start with hexagonal initialization (denser)
    xs, ys, rs = _initial_hexagonal()
    xs, ys, rs, score = _local_search(xs, ys, rs, deadline)
    if score > best_score:
        best_score = score
        best_xs, best_ys, best_rs = xs.copy(), ys.copy(), rs.copy()
    
    # Smooth polish
    if time.time() < deadline - 10:
        xs, ys, rs, score = _polish_smooth(xs, ys, rs, deadline)
        if score > best_score:
            best_score = score
            best_xs, best_ys, best_rs = xs.copy(), ys.copy(), rs.copy()

    # Try grid initialization
    xs, ys, rs = _initial_grid()
    xs, ys, rs, score = _local_search(xs, ys, rs, deadline)
    if score > best_score:
        best_score = score
        best_xs, best_ys, best_rs = xs.copy(), ys.copy(), rs.copy()
    
    # Smooth polish
    if time.time() < deadline - 10:
        xs, ys, rs, score = _polish_smooth(xs, ys, rs, deadline)
        if score > best_score:
            best_score = score
            best_xs, best_ys, best_rs = xs.copy(), ys.copy(), rs.copy()

    # Try several random restarts
    while time.time() < deadline - 15.0:
        xs, ys, rs = _initial_random()
        xs, ys, rs, score = _local_search(xs, ys, rs, deadline)
        if score > best_score:
            best_score = score
            best_xs, best_ys, best_rs = xs.copy(), ys.copy(), rs.copy()
        
        # Smooth polish on best random result
        if time.time() < deadline - 10:
            xs, ys, rs, score = _polish_smooth(xs, ys, rs, deadline)
            if score > best_score:
                best_score = score
                best_xs, best_ys, best_rs = xs.copy(), ys.copy(), rs.copy()

    # Final high-precision polish on best solution
    if best_xs is not None and time.time() < deadline - 5:
        xs, ys, rs, score = _polish_smooth(best_xs, best_ys, best_rs, deadline, temp=0.005)
        if score > best_score:
            best_score = score
            best_xs, best_ys, best_rs = xs.copy(), ys.copy(), rs.copy()
        
        # One more local search pass
        xs, ys, rs, score = _local_search(xs, ys, rs, deadline)
        if score > best_score:
            best_score = score
            best_xs, best_ys, best_rs = xs.copy(), ys.copy(), rs.copy()

    # Final validation
    if best_xs is not None:
        if not _validate(best_xs, best_ys, best_rs):
            # Fallback to grid if validation fails
            xs, ys, rs = _initial_grid()
            best_xs, best_ys, best_rs = xs, ys, rs

    result = np.column_stack([best_xs, best_ys, best_rs])
    return result
