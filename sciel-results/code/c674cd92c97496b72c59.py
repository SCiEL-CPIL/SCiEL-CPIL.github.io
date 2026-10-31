import numpy as np
from scipy.optimize import minimize
import time
import random

N = 21
DEADLINE = 140.0  # safety margin before 150s kill

def _rect_from_width(width):
    """Given rectangle width w, compute height h such that perimeter = 4."""
    # 2*(w + h) = 4 => h = 2 - w
    return 2.0 - width

def _evaluate(circles):
    """
    circles: (N, 3) array of [x, y, r]
    Returns (score, valid) where score = sum of radii if valid, else 0.
    """
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    # All radii must be positive
    if np.any(rs <= 1e-12):
        return 0.0, False
    
    # Determine bounding rectangle: use the min/max of circle extents
    # But actually, the problem says "bounding rectangle has perimeter at most 4"
    # The bounding rectangle is the smallest rectangle containing all circles
    # So its width = max(x+r) - min(x-r), height = max(y+r) - min(y-r)
    
    xmin = np.min(xs - rs)
    xmax = np.max(xs + rs)
    ymin = np.min(ys - rs)
    ymax = np.max(ys + rs)
    
    W = xmax - xmin
    H = ymax - ymin
    
    perim = 2.0 * (W + H)
    if perim > 4.0 + 1e-7:
        return 0.0, False
    
    # Check overlaps: distance between centers >= sum of radii
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            rad_sum = rs[i] + rs[j]
            if dist_sq < rad_sum * rad_sum - 1e-9:
                return 0.0, False
    
    return float(np.sum(rs)), True

def _objective(v, W):
    """
    Objective for optimization: v = [x1,y1,r1, ..., xN,yN,rN]
    W is fixed (rectangle width), H = 2 - W.
    We minimize -sum(r) + penalties.
    
    The circles must fit in rectangle [0, W] x [0, H].
    """
    H = _rect_from_width(W)
    circles = v.reshape(N, 3)
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    
    score = -np.sum(rs)
    penalty = 0.0
    
    # Boundary penalties: circles must be inside [0,W] x [0,H]
    for i in range(N):
        if xs[i] < rs[i]:
            penalty += (rs[i] - xs[i]) ** 2
        if xs[i] > W - rs[i]:
            penalty += (xs[i] - (W - rs[i])) ** 2
        if ys[i] < rs[i]:
            penalty += (rs[i] - ys[i]) ** 2
        if ys[i] > H - rs[i]:
            penalty += (ys[i] - (H - rs[i])) ** 2
    
    # Overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            rad_sum = rs[i] + rs[j]
            if dist < rad_sum:
                penalty += (rad_sum - dist) ** 2
    
    return score + 10.0 * penalty

def _grid_layout(W, H):
    """Create a grid layout of N circles in rectangle W x H."""
    cols = int(np.ceil(np.sqrt(N * W / H))) if W > 0 else int(np.ceil(np.sqrt(N)))
    rows = int(np.ceil(N / cols))
    cell_w = W / cols
    cell_h = H / rows
    max_r = min(cell_w, cell_h) * 0.45
    
    v = np.zeros(N * 3)
    for i in range(N):
        c = i % cols
        r = i // cols
        v[i*3] = (c + 0.5) * cell_w
        v[i*3+1] = (r + 0.5) * cell_h
        v[i*3+2] = max_r
    return v

def _hex_layout(W, H):
    """Create a hexagonal layout."""
    # Use a hexagonal pattern: offset every other row
    cols = int(np.ceil(np.sqrt(N * 2 * W / H))) if H > 0 else int(np.ceil(np.sqrt(N)))
    cell_w = W / cols
    cell_h = cell_w / np.sqrt(3)
    rows = int(np.ceil(N / cols))
    max_r = cell_w * 0.45
    
    v = np.zeros(N * 3)
    for i in range(N):
        c = i % cols
        r = i // cols
        offset = cell_w / 2 if r % 2 == 1 else 0.0
        x = (c + 0.5) * cell_w + offset
        # Clamp x
        x = np.clip(x, max_r, W - max_r)
        y = (r + 0.5) * cell_h
        y = np.clip(y, max_r, H - max_r)
        v[i*3] = x
        v[i*3+1] = y
        v[i*3+2] = max_r
    return v

def _random_layout(W, H, seed):
    """Random layout with small circles."""
    rng = random.Random(seed)
    v = np.zeros(N * 3)
    max_r = min(W, H) * 0.25
    for i in range(N):
        r = max_r * rng.uniform(0.5, 1.0)
        x = rng.uniform(r, W - r)
        y = rng.uniform(r, H - r)
        v[i*3] = x
        v[i*3+1] = y
        v[i*3+2] = r
    return v

def _optimize_from(v_init, W, time_budget):
    """Run SLSQP optimization from initial guess."""
    H = _rect_from_width(W)
    
    # Bounds: x in [0, W], y in [0, H], r in [1e-6, min(W,H)/2]
    r_max = min(W, H) * 0.5
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W))  # x
        bounds.append((1e-6, H))  # y
        bounds.append((1e-6, r_max))  # r
    
    start = time.time()
    try:
        res = minimize(
            _objective, v_init, args=(W,),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10}
        )
    except Exception:
        return v_init
    
    return res.x

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_score = 0.0
    best_circles = None
    
    # Try different aspect ratios (width of rectangle)
    # Perimeter = 4, so H = 2 - W
    # W ranges from small to 2 (but H must be positive, so W < 2)
    aspect_ratios = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8]
    
    # Also try square (W=1, H=1)
    # And some more
    
    # First pass: grid layouts for various W
    for W in aspect_ratios:
        if time.time() > deadline - 20:
            break
        H = _rect_from_width(W)
        if H <= 0:
            continue
        
        # Grid layout
        v_init = _grid_layout(W, H)
        v_opt = _optimize_from(v_init, W, 5.0)
        circles = v_opt.reshape(N, 3)
        score, valid = _evaluate(circles)
        if valid and score > best_score:
            best_score = score
            best_circles = circles.copy()
    
    # Second pass: hexagonal layouts
    for W in aspect_ratios:
        if time.time() > deadline - 15:
            break
        H = _rect_from_width(W)
        if H <= 0:
            continue
        
        v_init = _hex_layout(W, H)
        v_opt = _optimize_from(v_init, W, 5.0)
        circles = v_opt.reshape(N, 3)
        score, valid = _evaluate(circles)
        if valid and score > best_score:
            best_score = score
            best_circles = circles.copy()
    
    # Third pass: random restarts for best aspect ratios
    best_W_list = [1.0, 0.8, 1.2, 0.9, 1.1]
    for W in best_W_list:
        if time.time() > deadline - 10:
            break
        H = _rect_from_width(W)
        if H <= 0:
            continue
        
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            v_init = _random_layout(W, H, seed)
            v_opt = _optimize_from(v_init, W, 3.0)
            circles = v_opt.reshape(N, 3)
            score, valid = _evaluate(circles)
            if valid and score > best_score:
                best_score = score
                best_circles = circles.copy()
    
    # Fallback: simple grid in square
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        best_circles = np.zeros((N, 3))
        cell = 1.0 / side
        max_r = cell * 0.45
        for i in range(N):
            c = i % side
            r = i // side
            best_circles[i, 0] = (c + 0.5) * cell
            best_circles[i, 1] = (r + 0.5) * cell
            best_circles[i, 2] = max_r
    
    return best_circles

# EVOLVE-BLOCK-END
