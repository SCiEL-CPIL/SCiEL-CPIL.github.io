"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 21
TIME_LIMIT = 145.0  # safety margin before 150s kill

def _validate(arr):
    """Check if array of (x, y, r) is valid: disjoint, inside rect with perimeter <= 4."""
    if arr.shape != (N, 3):
        return False
    if np.any(arr[:, 2] <= 0):
        return False
    xs = arr[:, 0]
    ys = arr[:, 1]
    rs = arr[:, 2]
    left = np.min(xs - rs)
    right = np.max(xs + rs)
    top = np.max(ys + rs)
    bottom = np.min(ys - rs)
    W = right - left
    H = top - bottom
    if W <= 0 or H <= 0:
        return False
    if 2 * (W + H) > 4.0 + 1e-7:
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-7:
                return False
    return True

def _score(arr):
    """Compute the raw value: sum of radii, if valid."""
    if not _validate(arr):
        return 0.0
    return float(np.sum(arr[:, 2]))

def _separate_circles(x, y, r, W, H, iters=300, step_scale=0.001):
    """Force-directed separation to resolve overlaps."""
    n = len(x)
    for it in range(iters):
        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        # Circle-circle repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                min_d = r[i] + r[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    force = overlap * 0.5
                    fx[i] += (dx / d) * force
                    fy[i] += (dy / d) * force
                    fx[j] -= (dx / d) * force
                    fy[j] -= (dy / d) * force
        # Wall repulsion
        for i in range(n):
            if x[i] - r[i] < 0:
                fx[i] += (x[i] - r[i]) * 0.5
            if x[i] + r[i] > W:
                fx[i] -= (x[i] + r[i] - W) * 0.5
            if y[i] - r[i] < 0:
                fy[i] += (y[i] - r[i]) * 0.5
            if y[i] + r[i] > H:
                fy[i] -= (y[i] + r[i] - H) * 0.5
        # Apply forces
        step = step_scale * max(1.0, np.max(np.abs(fx)), np.max(np.abs(fy)))
        x += fx * step
        y += fy * step
        # Clamp to walls
        x = np.clip(x, r, W - r)
        y = np.clip(y, r, H - r)
    return x, y

def _compute_optimal_radii(x, y, W, H):
    """Compute the maximum uniform radius that fits given positions."""
    n = len(x)
    min_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            min_dist = min(min_dist, d)
    for i in range(n):
        min_dist = min(min_dist, x[i], W - x[i], y[i], H - y[i])
    return min_dist / 2

def _hex_initialization(W, H):
    """Create hexagonal packing initialization for N circles in W x H rectangle."""
    # Try to fit hexagonal rows
    # In hexagonal packing, row spacing = r * sqrt(3), col spacing = 2*r
    # With offset rows
    best_x = None
    best_y = None
    best_r = 0
    
    for n_rows in range(3, 8):
        for n_cols in range(3, 8):
            if n_rows * n_cols < N:
                continue
            # Try different offsets
            for offset in [0.0, 0.5]:
                positions = []
                for row in range(n_rows):
                    for col in range(n_cols):
                        if len(positions) >= N:
                            break
                        # Hexagonal offset
                        x_pos = (col + (offset if row % 2 else 0)) * 2.0
                        y_pos = row * np.sqrt(3)
                        positions.append((x_pos, y_pos))
                if len(positions) < N:
                    continue
                positions = positions[:N]
                xs = np.array([p[0] for p in positions])
                ys = np.array([p[1] for p in positions])
                # Normalize to fit in W x H
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                if x_max - x_min < 1e-9 or y_max - y_min < 1e-9:
                    continue
                xs_norm = (xs - x_min) / (x_max - x_min) * W
                ys_norm = (ys - y_min) / (y_max - y_min) * H
                # Compute max radius
                min_d = np.inf
                for i in range(N):
                    for j in range(i + 1, N):
                        d = np.sqrt((xs_norm[i]-xs_norm[j])**2 + (ys_norm[i]-ys_norm[j])**2)
                        min_d = min(min_d, d)
                for i in range(N):
                    min_d = min(min_d, xs_norm[i], W - xs_norm[i], ys_norm[i], H - ys_norm[i])
                r = min_d / 2
                if r > best_r:
                    best_r = r
                    best_x = xs_norm.copy()
                    best_y = ys_norm.copy()
    
    if best_x is None:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        positions = []
        for i in range(side):
            for j in range(side):
                if len(positions) >= N:
                    break
                positions.append(((i + 0.5) / side * W, (j + 0.5) / side * H))
        best_x = np.array([p[0] for p in positions])
        best_y = np.array([p[1] for p in positions])
        best_r = _compute_optimal_radii(best_x, best_y, W, H)
    
    return best_x, best_y, best_r

def _local_optimize(x, y, r, W, H, time_budget=5.0):
    """Local optimization using scipy minimize with penalty method."""
    n = N
    # Variables: x(n), y(n), r(n)
    def objective(v):
        xv = v[:n]
        yv = v[n:2*n]
        rv = v[2*n:3*n]
        if np.any(rv <= 0):
            return 1e10
        penalty = 0.0
        # Wall constraints
        for i in range(n):
            if xv[i] - rv[i] < 0:
                penalty += (xv[i] - rv[i]) ** 2
            if xv[i] + rv[i] > W:
                penalty += (xv[i] + rv[i] - W) ** 2
            if yv[i] - rv[i] < 0:
                penalty += (yv[i] - rv[i]) ** 2
            if yv[i] + rv[i] > H:
                penalty += (yv[i] + rv[i] - H) ** 2
        # Disjointness
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((xv[i]-xv[j])**2 + (yv[i]-yv[j])**2)
                min_d = rv[i] + rv[j]
                if d < min_d:
                    penalty += (min_d - d) ** 2
        return -np.sum(rv) + 100.0 * penalty
    
    v0 = np.concatenate([x, y, r])
    bounds = [(0, W)] * n + [(0, H)] * n + [(1e-6, min(W, H)/2)] * n
    
    t_start = time.time()
    try:
        result = minimize(objective, v0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'maxfun': 500, 'maxls': 20})
    except Exception:
        return x, y, r
    
    if result.success and result.fun < objective(v0):
        xv = result.x[:n]
        yv = result.x[n:2*n]
        rv = result.x[2*n:3*n]
        # Check validity
        valid = True
        for i in range(n):
            if xv[i] - rv[i] < -1e-6 or xv[i] + rv[i] > W + 1e-6:
                valid = False
                break
            if yv[i] - rv[i] < -1e-6 or yv[i] + rv[i] > H + 1e-6:
                valid = False
                break
        if valid:
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.sqrt((xv[i]-xv[j])**2 + (yv[i]-yv[j])**2)
                    if d < rv[i] + rv[j] - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            return xv, yv, rv
    
    return x, y, r

def _successive_convex_relaxation(x, y, r, W, H, max_iters=50, time_budget=10.0):
    """SCR: iteratively solve convex relaxations around current point."""
    n = N
    t_start = time.time()
    
    for it in range(max_iters):
        if time.time() - t_start > time_budget:
            break
        
        # Build linear constraints from current point
        # For each pair (i,j), linearize: |xi - xj| >= ri + rj
        # First-order Taylor: (xi-xj)(xi-xj)*0 + (yi-yj)(yi-yj)*0 >= ri+rj
        # Actually, use the distance constraint linearized:
        # dx*(xi-xj) + dy*(yi-yj) >= (ri+rj) * d where d = current distance
        
        # For walls: xi >= ri, xi <= W-ri, yi >= ri, yi <= H-ri
        
        # Use scipy.optimize.linprog
        # Variables: dx_i, dy_i, dr_i for each i (3n variables)
        # Minimize: -sum(dr_i)
        # Constraints:
        #   dx_i + x_i >= r_i + dr_i  =>  dx_i - dr_i >= r_i - x_i
        #   -dx_i + x_i <= W - r_i - dr_i  =>  -dx_i - dr_i <= W - r_i - x_i
        #   Similarly for y
        #   For pairs: linearized distance constraint
        
        # This is getting complex. Let's use a simpler approach:
        # Just do force-directed + radius recomputation iteratively
        
        # Force-directed separation
        x, y = _separate_circles(x, y, r, W, H, iters=100, step_scale=0.002)
        
        # Recompute optimal radius
        new_r = _compute_optimal_radii(x, y, W, H)
        if new_r > r[0] * (1 + 1e-6):
            r = np.full(n, new_r)
        else:
            break
    
    return x, y, r

def _high_precision_polish(x, y, r, W, H, time_budget=5.0):
    """Final polish with tight tolerances."""
    n = N
    t_start = time.time()
    
    # Use very small steps for fine adjustment
    for _ in range(200):
        if time.time() - t_start > time_budget:
            break
        improved = False
        for i in range(n):
            for dx in [-0.001, 0.001]:
                for dy in [-0.001, 0.001]:
                    new_x = x[i] + dx
                    new_y = y[i] + dy
                    if new_x - r[i] < 0 or new_x + r[i] > W:
                        continue
                    if new_y - r[i] < 0 or new_y + r[i] > H:
                        continue
                    valid = True
                    for j in range(n):
                        if j == i:
                            continue
                        d = np.sqrt((new_x-x[j])**2 + (new_y-y[j])**2)
                        if d < r[i] + r[j] - 1e-9:
                            valid = False
                            break
                    if valid:
                        x[i] = new_x
                        y[i] = new_y
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    # Recompute optimal radius
    new_r = _compute_optimal_radii(x, y, W, H)
    if new_r > r[0]:
        r = np.full(n, new_r)
    
    return x, y, r

def construct():
    t0 = time.time()
    
    best_arr = None
    best_score = 0.0
    
    # Try multiple aspect ratios
    aspect_ratios = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0]
    
    for ar in aspect_ratios:
        if time.time() - t0 > TIME_LIMIT - 15:
            break
        
        W = 2 * ar / (ar + 1)
        H = 2 / (ar + 1)
        
        # Hexagonal initialization
        x, y, r = _hex_initialization(W, H)
        
        # Force-directed separation
        x, y = _separate_circles(x, y, r, W, H, iters=200, step_scale=0.001)
        
        # Recompute optimal radius
        r = np.full(N, _compute_optimal_radii(x, y, W, H))
        
        arr = np.column_stack([x, y, r])
        s = _score(arr)
        if s > best_score:
            best_score = s
            best_arr = arr.copy()
        
        # Local optimization
        if time.time() - t0 > TIME_LIMIT - 20:
            continue
        
        x2, y2, r2 = _local_optimize(x, y, r, W, H, time_budget=3.0)
        arr2 = np.column_stack([x2, y2, r2])
        s2 = _score(arr2)
        if s2 > best_score:
            best_score = s2
            best_arr = arr2.copy()
        
        # SCR refinement
        if time.time() - t0 > TIME_LIMIT - 20:
            continue
        
        x3, y3, r3 = _successive_convex_relaxation(x2, y2, r2, W, H, max_iters=30, time_budget=5.0)
        arr3 = np.column_stack([x3, y3, r3])
        s3 = _score(arr3)
        if s3 > best_score:
            best_score = s3
            best_arr = arr3.copy()
    
    # Final high-precision polish on best solution
    if best_arr is not None and time.time() - t0 < TIME_LIMIT - 5:
        x = best_arr[:, 0].copy()
        y = best_arr[:, 1].copy()
        r = best_arr[:, 2].copy()
        # Find W, H from bounding box
        left = np.min(x - r)
        right = np.max(x + r)
        top = np.max(y + r)
        bottom = np.min(y - r)
        W = right - left
        H = top - bottom
        # Normalize to [0, W] x [0, H]
        x = x - left
        y = y - bottom
        
        x, y, r = _high_precision_polish(x, y, r, W, H, time_budget=5.0)
        arr = np.column_stack([x, y, r])
        s = _score(arr)
        if s > best_score:
            best_score = s
            best_arr = arr
    
    if best_arr is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_arr = np.array(cells[:N])
    
    return best_arr
