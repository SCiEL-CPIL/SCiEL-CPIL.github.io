"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 21

# EVOLVE-BLOCK-START

def _is_valid(x, y, r, tol=1e-6):
    """Check if circles are valid: non-overlapping and within perimeter-4 bounding box."""
    n = len(r)
    # Check non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if d < r[i] + r[j] - tol:
                return False
    
    # Check perimeter constraint
    x_min = np.min(x - r)
    x_max = np.max(x + r)
    y_min = np.min(y - r)
    y_max = np.max(y + r)
    bw = x_max - x_min
    bh = y_max - y_min
    perimeter = 2 * (bw + bh)
    if perimeter > 4 + tol:
        return False
    
    return True


def _force_relax(x, y, r, n_iter=200, learning_rate=0.01, max_time=100.0, start_time=None):
    """
    Force-directed relaxation to resolve overlaps and improve packing.
    Returns updated x, y, r.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(r)
    for it in range(n_iter):
        if time.time() - start_time > max_time:
            break
        
        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Repulsion between overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy) + 1e-10
                min_d = r[i] + r[j]
                if d < min_d:
                    # Repulsive force
                    force = (min_d - d) / d
                    fx[i] += force * dx
                    fy[i] += force * dy
                    fx[j] -= force * dx
                    fy[j] -= force * dy
        
        # Boundary forces: push circles toward center if they're at edges
        # Compute bounding box
        x_min = np.min(x - r)
        x_max = np.max(x + r)
        y_min = np.min(y - r)
        y_max = np.max(y + r)
        bw = x_max - x_min
        bh = y_max - y_min
        perimeter = 2 * (bw + bh)
        
        # If perimeter is too large, apply inward forces
        if perimeter > 4:
            excess = (perimeter - 4) / 4
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            for i in range(n):
                fx[i] -= excess * (x[i] - cx) * 0.1
                fy[i] -= excess * (y[i] - cy) * 0.1
        
        # Update positions
        x += learning_rate * fx
        y += learning_rate * fy
    
    return x, y


def _expand_radii(x, y, r, max_time=100.0, start_time=None):
    """
    Try to expand radii while maintaining validity.
    Returns updated r.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(r)
    step = 0.001
    for it in range(500):
        if time.time() - start_time > max_time:
            break
        
        # Try to expand each radius slightly
        improved = False
        for i in range(n):
            new_r = r.copy()
            new_r[i] += step
            if _is_valid(x, y, new_r):
                r = new_r
                improved = True
                break
        
        if not improved:
            # Try smaller step
            step *= 0.5
            if step < 1e-6:
                break
    
    return r


def _slsqp_optimize(x0, y0, r0, max_time=100.0, start_time=None):
    """
    Use SLSQP to optimize the packing.
    Returns (x, y, r) or None if failed.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(r0)
    
    def objective(vars_):
        if time.time() - start_time > max_time:
            return 1e10
        x = vars_[:n]
        y = vars_[n:2*n]
        r = vars_[2*n:3*n]
        
        # Penalty for perimeter > 4
        x_min = np.min(x - r)
        x_max = np.max(x + r)
        y_min = np.min(y - r)
        y_max = np.max(y + r)
        bw = x_max - x_min
        bh = y_max - y_min
        perim = 2 * (bw + bh)
        perim_pen = max(0, perim - 4) ** 2 * 1000
        
        # Penalty for overlap
        overlap_pen = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                min_d = r[i] + r[j]
                if d < min_d:
                    overlap_pen += (min_d - d)**2 * 500
        
        return -np.sum(r) + perim_pen + overlap_pen
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0, 2.0))  # x
    for i in range(n):
        bounds.append((0, 2.0))  # y
    for i in range(n):
        bounds.append((0.01, 0.5))  # r
    
    try:
        result = minimize(
            objective,
            np.concatenate([x0, y0, r0]),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-6}
        )
        
        if result.success or (time.time() - start_time < max_time):
            x_opt = result.x[:n]
            y_opt = result.x[n:2*n]
            r_opt = result.x[2*n:3*n]
            return x_opt, y_opt, r_opt
    except Exception:
        pass
    
    return None


def _polish_result(x, y, r, max_time=100.0, start_time=None):
    """
    Polish the result with force relaxation and radius expansion.
    Returns (x, y, r).
    """
    if start_time is None:
        start_time = time.time()
    
    # Force relaxation
    x, y = _force_relax(x, y, r, n_iter=100, learning_rate=0.005, max_time=max_time, start_time=start_time)
    
    # Expand radii
    r = _expand_radii(x, y, r, max_time=max_time, start_time=start_time)
    
    # SLSQP optimization
    if time.time() - start_time < max_time - 5:
        result = _slsqp_optimize(x, y, r, max_time=max_time, start_time=start_time)
        if result is not None:
            x, y, r = result
    
    # Final polish
    x, y = _force_relax(x, y, r, n_iter=50, learning_rate=0.002, max_time=max_time, start_time=start_time)
    r = _expand_radii(x, y, r, max_time=max_time, start_time=start_time)
    
    return x, y, r


def _grid_placement(n, aspect_ratio=1.0):
    """
    Place circles in a grid pattern.
    Returns (x, y, r).
    """
    cols = int(np.ceil(np.sqrt(n * aspect_ratio)))
    rows = int(np.ceil(n / cols))
    
    # For perimeter 4: w + h = 2
    # Let's use a rectangle with aspect ratio
    h = 1.0
    w = 2.0 - h  # so w + h = 2
    
    # Scale to fit
    cell_w = w / cols
    cell_h = h / rows
    
    xs = []
    ys = []
    rs = []
    count = 0
    for i in range(cols):
        for j in range(rows):
            if count >= n:
                break
            x = (i + 0.5) * cell_w
            y = (j + 0.5) * cell_h
            r = min(cell_w, cell_h) * 0.45
            xs.append(x)
            ys.append(y)
            rs.append(r)
            count += 1
    
    return np.array(xs[:n]), np.array(ys[:n]), np.array(rs[:n])


def _random_placement(n, aspect_ratio=1.0):
    """
    Random placement of circles.
    Returns (x, y, r).
    """
    h = 1.0
    w = 2.0 - h
    
    xs = np.random.rand(n) * w
    ys = np.random.rand(n) * h
    rs = np.random.rand(n) * 0.1 + 0.05
    
    return xs, ys, rs


def _pack_circles(n, max_time=100.0):
    """
    Pack n circles in a rectangle of perimeter <= 4 to maximize sum of radii.
    Returns array of shape (n, 3) with columns [x, y, r].
    """
    start_time = time.time()
    
    best_result = None
    best_value = -np.inf
    
    def _evaluate(x, y, r):
        """Check validity and compute objective."""
        if time.time() - start_time > max_time:
            return None, -np.inf
        
        if not _is_valid(x, y, r):
            return None, -np.inf
        
        value = np.sum(r)
        return np.column_stack([x, y, r]), value
    
    # Try multiple strategies
    strategies = [
        ("grid", lambda: _grid_placement(n, aspect_ratio=1.0)),
        ("grid_wide", lambda: _grid_placement(n, aspect_ratio=1.5)),
        ("grid_tall", lambda: _grid_placement(n, aspect_ratio=0.7)),
        ("random", lambda: _random_placement(n, aspect_ratio=1.0)),
        ("random2", lambda: _random_placement(n, aspect_ratio=1.0)),
        ("random3", lambda: _random_placement(n, aspect_ratio=1.0)),
    ]
    
    for strategy_name, placement_func in strategies:
        if time.time() - start_time > max_time - 10:
            break
        
        try:
            x, y, r = placement_func()
            
            # Polish
            x, y, r = _polish_result(x, y, r, max_time=max_time, start_time=start_time)
            
            # Evaluate
            valid_result, value = _evaluate(x, y, r)
            if valid_result is not None and value > best_value:
                best_value = value
                best_result = valid_result
        except Exception:
            continue
    
    if best_result is None:
        # Fallback: return a simple valid configuration
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result


def construct():
    """
    Returns array of shape (n, 3): n disjoint circles whose bounding rectangle has perimeter at most 4.
    """
    result = _pack_circles(N, max_time=140.0)
    return result

# EVOLVE-BLOCK-END
