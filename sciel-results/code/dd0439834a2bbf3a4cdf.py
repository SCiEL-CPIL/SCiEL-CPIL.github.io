"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
N = 21
import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def _validate(circles):
    """Check circles are valid: disjoint interiors, inside rectangle, perimeter <= 4.
    Returns (valid, perimeter, sum_radii)."""
    if len(circles) != N:
        return False, float('inf'), 0.0
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    if np.any(rs <= 0):
        return False, float('inf'), 0.0
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    w = max_x - min_x
    h = max_y - min_y
    perim = 2.0 * (w + h)
    if perim > 4.0 + 1e-7:
        return False, perim, 0.0
    # Check pairwise disjoint
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - 1e-7:
                return False, perim, 0.0
    return True, perim, float(np.sum(rs))

def _grid_init(n, aspect):
    """Create a grid-like initialization for n circles in a rectangle with given aspect ratio.
    aspect = w/h. Perimeter constraint: 2(w+h)=4, so w+h=2.
    w = aspect*h, so aspect*h + h = 2, h = 2/(aspect+1), w = 2*aspect/(aspect+1)."""
    cols = int(np.ceil(np.sqrt(n * aspect)))
    rows = int(np.ceil(n / cols))
    h = 2.0 / (aspect + 1.0)
    w = aspect * h
    # Shift so bounding box starts at origin
    circles = []
    for idx in range(n):
        r = idx % cols
        c = idx // cols
        # Center of cell (r, c) in grid of cols x rows
        cx = (r + 0.5) * w / cols
        cy = (c + 0.5) * h / rows
        # Max radius limited by cell size and distance to neighbors
        cell_w = w / cols
        cell_h = h / rows
        max_r = min(cell_w, cell_h) / 2.0 * 0.95
        circles.append([cx, cy, max_r])
    return np.array(circles)

def _objective(circles, aspect):
    """Penalty-based objective for optimization. We want to maximize sum of radii
    subject to constraints. We'll use a penalty method."""
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    # Bounding box
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    w = max_x - min_x
    h = max_y - min_y
    perim = 2.0 * (w + h)
    
    # Perimeter penalty
    perim_pen = max(0.0, perim - 4.0) ** 2 * 100.0
    
    # Overlap penalty
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = rs[i] + rs[j] - dist
            if overlap > 0:
                overlap_pen += overlap ** 2 * 100.0
    
    # Radius positivity penalty
    rad_pen = 0.0
    for i in range(N):
        if rs[i] < 0.01:
            rad_pen += (0.01 - rs[i]) ** 2 * 10.0
    
    # Objective: maximize sum of radii, minimize penalties
    val = float(np.sum(rs)) - perim_pen - overlap_pen - rad_pen
    return val

def _local_optimize(circles, max_iter=200, tol=1e-8):
    """Simple gradient-free local optimization using coordinate descent + radius growth."""
    best = circles.copy()
    best_val = _objective(best, 1.0)
    
    for it in range(max_iter):
        improved = False
        # Try to grow radii
        for i in range(N):
            old_r = best[i, 2]
            # Try increasing radius
            new_r = old_r * 1.02
            if new_r < 0.5:  # Cap radius
                trial = best.copy()
                trial[i, 2] = new_r
                val = _objective(trial, 1.0)
                if val > best_val + 1e-10:
                    best = trial
                    best_val = val
                    improved = True
        
        # Try moving circles
        for i in range(N):
            for axis in [0, 1]:
                # Try small moves
                for delta in [0.01, -0.01, 0.005, -0.005]:
                    trial = best.copy()
                    trial[i, axis] += delta
                    val = _objective(trial, 1.0)
                    if val > best_val + 1e-10:
                        best = trial
                        best_val = val
                        improved = True
        
        if not improved:
            break
    
    return best

def _nelder_mead(func, x0, bounds=None, max_iter=1000, tol=1e-6):
    """Simple Nelder-Mead implementation."""
    n = len(x0)
    simplex = [x0.copy()]
    for i in range(n):
        x = x0.copy()
        x[i] += 0.1 if x0[i] >= 0 else -0.1
        simplex.append(x)
    
    fvals = [func(x) for x in simplex]
    
    for it in range(max_iter):
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        alpha = 1.0
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = func(xr)
        
        if fr < fvals[0]:
            # Expansion
            gamma = 2.0
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            rho = 0.5
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = func(xc)
            if fc < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = func(simplex[i])
    
    return min(simplex, key=lambda x: func(x)), min(fvals)

def _flatten(circles):
    """Flatten circles array to 1D for optimization."""
    return circles.flatten()

def _unflatten(x):
    """Unflatten 1D array to circles array."""
    return x.reshape(N, 3)

def _penalty_objective(x, aspect):
    """Objective function for Nelder-Mead: minimize negative sum of radii + penalties."""
    circles = _unflatten(x)
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    # Ensure positive radii
    rs = np.maximum(rs, 0.001)
    
    # Bounding box
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    w = max_x - min_x
    h = max_y - min_y
    perim = 2.0 * (w + h)
    
    # Perimeter penalty (heavy)
    perim_pen = max(0.0, perim - 4.0) ** 2 * 50.0
    
    # Overlap penalty (heavy)
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = rs[i] + rs[j] - dist
            if overlap > 0:
                overlap_pen += overlap ** 2 * 50.0
    
    # Objective: minimize negative sum of radii + penalties
    val = -float(np.sum(rs)) + perim_pen + overlap_pen
    return val

def _optimize_for_aspect(aspect, timeout_check=None):
    """Optimize circle packing for a given aspect ratio."""
    h = 2.0 / (aspect + 1.0)
    w = aspect * h
    
    # Try different grid configurations
    best_circles = None
    best_val = -float('inf')
    
    for cols in range(3, 8):
        rows = int(np.ceil(N / cols))
        if rows < 3:
            continue
        
        # Initialize grid
        circles = np.zeros((N, 3))
        for idx in range(N):
            r = idx % cols
            c = idx // cols
            cx = (r + 0.5) * w / cols
            cy = (c + 0.5) * h / rows
            cell_w = w / cols
            cell_h = h / rows
            max_r = min(cell_w, cell_h) / 2.0 * 0.9
            circles[idx] = [cx, cy, max_r]
        
        # Run Nelder-Mead on flattened parameters
        x0 = circles.flatten()
        
        def obj(x):
            if timeout_check is not None and timeout_check():
                return 1e10
            return _penalty_objective(x, aspect)
        
        x_opt, f_opt = _nelder_mead(obj, x0, max_iter=300, tol=1e-5)
        
        circles_opt = _unflatten(x_opt)
        # Clamp radii to positive
        circles_opt[:, 2] = np.maximum(circles_opt[:, 2], 0.001)
        
        valid, perim, sum_r = _validate(circles_opt)
        if valid:
            if sum_r > best_val:
                best_val = sum_r
                best_circles = circles_opt
    
    return best_circles, best_val

def construct():
    """Main entry point: find optimal circle packing."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    def timeout_check():
        return time.time() > deadline
    
    best_circles = None
    best_val = -float('inf')
    
    # Try multiple aspect ratios
    aspects = [1.0, 0.8, 1.2, 0.7, 1.4, 0.6, 1.5, 0.5, 1.6, 0.4, 2.0, 0.3, 2.5, 0.25, 3.0]
    
    for aspect in aspects:
        if timeout_check():
            break
        
        circles, val = _optimize_for_aspect(aspect, timeout_check)
        
        if circles is not None and val > best_val:
            best_val = val
            best_circles = circles
        
        # Check time
        if timeout_check():
            break
    
    # If no valid solution found, use grid fallback
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    # Final validation
    valid, perim, sum_r = _validate(best_circles)
    if not valid:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles
# EVOLVE-BLOCK-END
