"""Improved program for circle packing in perimeter-4 rectangle."""
import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 21


def _validate(circles):
    """Check if circles are valid: disjoint interiors, inside rectangle with perimeter <= 4."""
    if circles is None or len(circles) != N:
        return False, 0.0
    
    # Compute bounding box
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    x_min = xs - rs
    x_max = xs + rs
    y_min = ys - rs
    y_max = ys + rs
    
    W = x_max.max() - x_min.min()
    H = y_max.max() - y_min.min()
    
    if W <= 0 or H <= 0:
        return False, 0.0
    
    perimeter = 2 * (W + H)
    if perimeter > 4 + 1e-9:
        return False, 0.0
    
    # Check pairwise non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            min_dist = rs[i] + rs[j]
            if dist_sq < min_dist * min_dist - 1e-12:
                return False, 0.0
    
    return True, rs.sum()


def _objective(params, W, H):
    """
    Objective: maximize sum of radii.
    params: flattened array [x0, y0, r0, x1, y1, r1, ...]
    Constraint: circles must be inside [0, W] x [0, H] and non-overlapping.
    We minimize negative sum of radii with penalty for violations.
    """
    n = N
    xs = params[0:3*n:3]
    ys = params[1:3*n:3]
    rs = params[2:3*n:3]
    
    # Ensure positive radii
    rs = np.maximum(rs, 1e-6)
    
    # Boundary violations
    bound_pen = 0.0
    for i in range(n):
        if xs[i] < rs[i]:
            bound_pen += (rs[i] - xs[i]) ** 2
        if xs[i] + rs[i] > W:
            bound_pen += (xs[i] + rs[i] - W) ** 2
        if ys[i] < rs[i]:
            bound_pen += (rs[i] - ys[i]) ** 2
        if ys[i] + rs[i] > H:
            bound_pen += (ys[i] + rs[i] - H) ** 2
    
    # Overlap violations
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                overlap_pen += (min_dist - dist) ** 2
    
    # Perimeter constraint: 2*(W+H) <= 4, so W+H <= 2
    # W and H are fixed in this formulation; we handle perimeter via the rectangle choice
    
    penalty = 1000.0 * (bound_pen + overlap_pen)
    
    # We want to maximize sum(rs), so minimize -sum(rs)
    return -rs.sum() + penalty


def _construct_for_aspect(aspect_ratio, deadline):
    """
    Try to pack circles in a rectangle of given aspect ratio (W/H).
    Perimeter constraint: 2*(W+H) <= 4 => W+H <= 2.
    If H = 1, W = aspect_ratio, then W+H = aspect_ratio + 1 <= 2 => aspect_ratio <= 1.
    If W = 1, H = 1/aspect_ratio, then W+H = 1 + 1/aspect_ratio <= 2 => aspect_ratio >= 1.
    
    Let's parametrize by H. If aspect = W/H, then W = aspect * H.
    2*(W+H) = 2*H*(aspect+1) <= 4 => H <= 2/(aspect+1).
    Set H = 2/(aspect+1) to maximize the rectangle (tight perimeter).
    W = aspect * H = 2*aspect/(aspect+1).
    """
    H = 2.0 / (aspect_ratio + 1.0)
    W = aspect_ratio * H
    
    # Initial guess: grid layout
    # For 21 circles, try various grid arrangements
    best_result = None
    best_score = -np.inf
    
    # Try different initial configurations
    configs = []
    
    # Grid-like inits
    for cols in range(3, 8):
        rows = int(np.ceil(N / cols))
        if rows * cols >= N:
            # Place circles in a grid
            xs = []
            ys = []
            rs = []
            for i in range(cols):
                for j in range(rows):
                    if len(xs) >= N:
                        break
                    x = (i + 0.5) * W / cols
                    y = (j + 0.5) * H / rows
                    r = min(W / cols, H / rows) / 2 * 0.95
                    xs.append(x)
                    ys.append(y)
                    rs.append(r)
            if len(xs) == N:
                configs.append((np.array(xs), np.array(ys), np.array(rs)))
    
    # Hexagonal-like init
    # Try to arrange in rows with offset
    row_counts = [4, 4, 4, 4, 3, 2]  # 21 circles
    if sum(row_counts) == N:
        xs = []
        ys = []
        rs = []
        row_h = H / (len(row_counts) + 0.5)
        for row_idx, count in enumerate(row_counts):
            y = (row_idx + 1.0) * row_h
            if row_idx % 2 == 0:
                for i in range(count):
                    x = (i + 0.5) * W / count
                    xs.append(x)
                    ys.append(y)
                    rs.append(min(W / max(count, 1), row_h) / 2 * 0.95)
            else:
                for i in range(count):
                    x = (i + 0.5) * W / count
                    xs.append(x)
                    ys.append(y)
                    rs.append(min(W / max(count, 1), row_h) / 2 * 0.95)
        if len(xs) == N:
            configs.append((np.array(xs), np.array(ys), np.array(rs)))
    
    # Random inits
    np.random.seed(42)
    for _ in range(5):
        xs = np.random.uniform(0.1 * W, 0.9 * W, N)
        ys = np.random.uniform(0.1 * H, 0.9 * H, N)
        rs = np.full(N, 0.05 * min(W, H))
        configs.append((xs, ys, rs))
    
    for xs_init, ys_init, rs_init in configs:
        if time.time() > deadline:
            break
        
        params_init = np.concatenate([xs_init, ys_init, rs_init])
        
        # Use L-BFGS-B with bounds
        bounds = []
        for i in range(N):
            bounds.append((0, W))  # x
            bounds.append((0, H))  # y
            bounds.append((1e-6, min(W, H) / 2))  # r
        
        try:
            result = minimize(
                _objective,
                params_init,
                args=(W, H),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_score:
                params = result.x
                xs = params[0:3*N:3]
                ys = params[1:3*N:3]
                rs = params[2:3*N:3]
                
                # Check validity
                circles = np.column_stack([xs, ys, rs])
                valid, score = _validate(circles)
                if valid and score > best_score:
                    best_score = score
                    best_result = circles.copy()
                    
        except Exception:
            pass
    
    return best_result, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_circles = None
    best_score = -np.inf
    
    # Try various aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.43, 0.6, 1.67, 0.5, 2.0, 0.4]
    
    for aspect in aspect_ratios:
        if time.time() > deadline:
            break
        
        circles, score = _construct_for_aspect(aspect, deadline)
        if circles is not None and score > best_score:
            best_score = score
            best_circles = circles.copy()
    
    # If no valid result found, return a simple grid
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles
