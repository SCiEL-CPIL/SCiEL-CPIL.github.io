"""Circle packing in perimeter-4 rectangle using physical simulation approach."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _is_valid(x, y, r, tol=1e-5):
    """Check validity: non-overlapping circles within perimeter-4 bounding box."""
    n = len(r)
    # Check non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx*dx + dy*dy
            min_d = r[i] + r[j]
            if d2 < min_d*min_d - tol:
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


def _physical_simulate(x0, y0, r0, rect_w, rect_h, max_time=30.0, start_time=None):
    """
    Physical simulation: circles with corner bias, collision resolution.
    Uses symplectic-like integration with damping.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(r0)
    x = x0.copy().astype(float)
    y = y0.copy().astype(float)
    r = r0.copy().astype(float)
    
    dt = 0.01
    damping = 0.95
    corner_force = 0.3
    collision_strength = 5.0
    center_force = 0.1
    
    # Corners of rectangle
    corners = [(0, 0), (rect_w, 0), (0, rect_h), (rect_w, rect_h)]
    
    for step in range(2000):
        if time.time() - start_time > max_time:
            break
        
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Corner attraction: bias toward nearest corner
        for i in range(n):
            # Find nearest corner
            dists = [(x[i] - cx)**2 + (y[i] - cy)**2 for cx, cy in corners]
            nearest = corners[np.argmin(dists)]
            cx, cy = nearest
            dx = cx - x[i]
            dy = cy - y[i]
            d = np.sqrt(dx*dx + dy*dy) + 1e-10
            # Force proportional to distance, but not too strong
            f = corner_force * 0.5
            fx[i] += f * dx / d
            fy[i] += f * dy / d
        
        # Center attraction to prevent all going to corners
        cx = rect_w / 2
        cy = rect_h / 2
        for i in range(n):
            fx[i] += center_force * (cx - x[i]) * 0.1
            fy[i] += center_force * (cy - y[i]) * 0.1
        
        # Collision forces between circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx*dx + dy*dy
                d = np.sqrt(d2) + 1e-10
                min_d = r[i] + r[j]
                if d < min_d:
                    overlap = min_d - d
                    force = collision_strength * overlap / d
                    fx[i] += force * dx
                    fy[i] += force * dy
                    fx[j] -= force * dx
                    fy[j] -= force * dy
        
        # Boundary forces: keep circles inside rectangle
        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += collision_strength * (r[i] - x[i])
            if x[i] + r[i] > rect_w:
                fx[i] -= collision_strength * (x[i] + r[i] - rect_w)
            if y[i] - r[i] < 0:
                fy[i] += collision_strength * (r[i] - y[i])
            if y[i] + r[i] > rect_h:
                fy[i] -= collision_strength * (y[i] + r[i] - rect_h)
        
        # Velocity update (implicit damping)
        x = x + dt * fx * damping
        y = y + dt * fy * damping
        
        # Hard constraint: ensure circles stay within bounds
        for i in range(n):
            x[i] = np.clip(x[i], r[i], rect_w - r[i])
            y[i] = np.clip(y[i], r[i], rect_h - r[i])
        
        # If perimeter would exceed 4, scale down positions toward center
        x_min = np.min(x - r)
        x_max = np.max(x + r)
        y_min = np.min(y - r)
        y_max = np.max(y + r)
        bw = x_max - x_min
        bh = y_max - y_min
        if 2*(bw + bh) > 4:
            # Scale positions to fit
            scale = 2.0 / (bw + bh)
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            for i in range(n):
                x[i] = cx + (x[i] - cx) * scale
                y[i] = cy + (y[i] - cy) * scale
    
    return x, y, r


def _expand_radii_sim(x, y, r, rect_w, rect_h, max_time=10.0, start_time=None):
    """Gradually expand radii while maintaining validity via simulation."""
    if start_time is None:
        start_time = time.time()
    
    n = len(r)
    current_sum = np.sum(r)
    
    for it in range(200):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        # Try expanding each circle slightly
        for i in range(n):
            trial_r = r.copy()
            trial_r[i] += 0.002
            # Check if we can still fit
            x_min = np.min(x - trial_r)
            x_max = np.max(x + trial_r)
            y_min = np.min(y - trial_r)
            y_max = np.max(y + trial_r)
            if 2*((x_max-x_min) + (y_max-y_min)) <= 4.01:
                # Check overlaps
                valid = True
                for j in range(n):
                    if j == i:
                        continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < trial_r[i] + r[j] - 1e-6:
                        valid = False
                        break
                if valid:
                    r = trial_r
                    improved = True
                    break
        
        if not improved:
            break
    
    return r


def _slsqp_refine(x, y, r, rect_w, rect_h, max_time=20.0, start_time=None):
    """SLSQP optimization to refine positions and radii."""
    if start_time is None:
        start_time = time.time()
    
    n = len(r)
    
    def objective(v):
        if time.time() - start_time > max_time:
            return 1e10
        x = v[:n]
        y = v[n:2*n]
        r = np.maximum(v[2*n:3*n], 0.01)
        
        # Objective: maximize sum of radii (negative for minimization)
        obj = -np.sum(r)
        
        # Perimeter penalty
        x_min = np.min(x - r)
        x_max = np.max(x + r)
        y_min = np.min(y - r)
        y_max = np.max(y + r)
        perim = 2*((x_max - x_min) + (y_max - y_min))
        if perim > 4:
            obj += (perim - 4)**2 * 500
        
        # Overlap penalty
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                min_d = r[i] + r[j]
                if d < min_d:
                    obj += (min_d - d)**2 * 200
        
        return obj
    
    # Constraints: keep within rectangle
    cons = []
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - np.maximum(v[2*n+i], 0.01)})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[n+i] - np.maximum(v[2*n+i], 0.01)})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: (rect_w - v[2*n+i]) - v[i]})
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: (rect_h - v[2*n+i]) - v[n+i]})
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, rect_w))
    for i in range(n):
        bounds.append((0.0, rect_h))
    for i in range(n):
        bounds.append((0.01, 0.8))
    
    try:
        v0 = np.concatenate([x, y, r])
        result = minimize(objective, v0, method='SLSQP', bounds=bounds, 
                          options={'maxiter': 100, 'ftol': 1e-7})
        if result.success:
            x_opt = result.x[:n]
            y_opt = result.x[n:2*n]
            r_opt = np.maximum(result.x[2*n:3*n], 0.01)
            return x_opt, y_opt, r_opt
    except Exception:
        pass
    
    return x, y, r


def _grid_init(n, w, h):
    """Grid initialization for circles."""
    cols = int(np.ceil(np.sqrt(n * w / h)))
    rows = int(np.ceil(n / cols))
    cell_w = w / cols
    cell_h = h / rows
    
    xs, ys, rs = [], [], []
    count = 0
    for i in range(cols):
        for j in range(rows):
            if count >= n:
                break
            x = (i + 0.5) * cell_w
            y = (j + 0.5) * cell_h
            r = min(cell_w, cell_h) * 0.42
            xs.append(x)
            ys.append(y)
            rs.append(r)
            count += 1
    
    return np.array(xs[:n]), np.array(ys[:n]), np.array(rs[:n])


def _random_init(n, w, h, seed=None):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    xs = rng.rand(n) * (w - 0.1) + 0.05
    ys = rng.rand(n) * (h - 0.1) + 0.05
    rs = rng.rand(n) * 0.08 + 0.04
    return xs, ys, rs


def _polish(x, y, r, rect_w, rect_h, max_time=30.0, start_time=None):
    """Full polishing pipeline: simulation + expansion + SLSQP."""
    if start_time is None:
        start_time = time.time()
    
    # Step 1: Physical simulation
    t1 = time.time()
    x, y, r = _physical_simulate(x, y, r, rect_w, rect_h, 
                                  max_time=min(15.0, max_time - (t1 - start_time)),
                                  start_time=start_time)
    
    # Step 2: Expand radii
    t2 = time.time()
    r = _expand_radii_sim(x, y, r, rect_w, rect_h,
                          max_time=min(8.0, max_time - (t2 - start_time)),
                          start_time=start_time)
    
    # Step 3: SLSQP refinement
    t3 = time.time()
    x, y, r = _slsqp_refine(x, y, r, rect_w, rect_h,
                            max_time=min(15.0, max_time - (t3 - start_time)),
                            start_time=start_time)
    
    # Step 4: Final expansion
    t4 = time.time()
    r = _expand_radii_sim(x, y, r, rect_w, rect_h,
                          max_time=min(5.0, max_time - (t4 - start_time)),
                          start_time=start_time)
    
    return x, y, r


def _try_configuration(w, h, init_type, seed, max_time=30.0, start_time=None):
    """Try one configuration: init + polish + evaluate."""
    if start_time is None:
        start_time = time.time()
    
    if init_type == "grid":
        x, y, r = _grid_init(N, w, h)
    else:
        x, y, r = _random_init(N, w, h, seed=seed)
    
    x, y, r = _polish(x, y, r, w, h, max_time=max_time, start_time=start_time)
    
    # Evaluate
    if _is_valid(x, y, r):
        value = np.sum(r)
        return np.column_stack([x, y, r]), value
    else:
        return None, -np.inf


def construct():
    """
    Returns array of shape (n, 3): n disjoint circles whose bounding rectangle has perimeter at most 4.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_result = None
    best_value = -np.inf
    
    # Different rectangle configurations (w + h = 2 for perimeter 4)
    configs = [
        # (width, height, init_type, seed)
        (1.0, 1.0, "grid", None),
        (1.5, 0.5, "grid", None),
        (0.5, 1.5, "grid", None),
        (1.2, 0.8, "grid", None),
        (0.8, 1.2, "grid", None),
        (1.0, 1.0, "random", 42),
        (1.5, 0.5, "random", 123),
        (1.2, 0.8, "random", 456),
        (1.0, 1.0, "random", 789),
        (0.9, 1.1, "random", 1011),
    ]
    
    for w, h, init_type, seed in configs:
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        try:
            result, value = _try_configuration(w, h, init_type, seed, 
                                                max_time=min(28.0, remaining - 2),
                                                start_time=start_time)
            if result is not None and value > best_value:
                best_value = value
                best_result = result
        except Exception:
            continue
    
    # Fallback if nothing valid found
    if best_result is None:
        # Simple grid in 1x1 square
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append([x, y, r])
        best_result = np.array(cells[:N])
    
    return best_result

# EVOLVE-BLOCK-END
