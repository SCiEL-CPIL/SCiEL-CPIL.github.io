"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 circles."""
import time
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = 140.0  # safety margin before 150s kill
START_TIME = time.time()


def elapsed():
    return time.time() - START_TIME


def is_valid(circles, tol=1e-9):
    """Check if circles are valid: inside bounding box with perimeter <= 4, no overlaps."""
    n = len(circles)
    if n == 0:
        return False
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    # Bounding box
    x_min = np.min(xs - rs)
    x_max = np.max(xs + rs)
    y_min = np.min(ys - rs)
    y_max = np.max(ys + rs)
    width = x_max - x_min
    height = y_max - y_min
    perimeter = 2 * (width + height)
    if perimeter > 4.0 + tol:
        return False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    return True


def objective(params, aspect_ratio):
    """
    params: 2N values = (x_0, y_0, ..., x_{N-1}, y_{N-1}, r_0, r_1, ..., r_{N-1})
    We fix the rectangle to have aspect_ratio = width/height.
    Perimeter = 2*(w+h) = 4, so h = 2/(1+ar), w = 2*ar/(1+ar).
    We scale positions to fit in [0, w] x [0, h].
    Actually, let's work in normalized coordinates [0,1]x[0,1] then scale.
    Better: work directly in physical coordinates.
    """
    n = N
    # Extract positions and radii
    xs = params[0:n]
    ys = params[n:2*n]
    rs = params[2*n:3*n]
    
    # Rectangle dimensions
    ar = aspect_ratio
    h = 2.0 / (1.0 + ar)
    w = 2.0 * ar / (1.0 + ar)
    
    # Penalty for going outside the rectangle
    penalty = 0.0
    for i in range(n):
        # Circle must be inside [0,w] x [0,h]
        if xs[i] - rs[i] < 0:
            penalty += (xs[i] - rs[i])**2
        if xs[i] + rs[i] > w:
            penalty += (xs[i] + rs[i] - w)**2
        if ys[i] - rs[i] < 0:
            penalty += (ys[i] - rs[i])**2
        if ys[i] + rs[i] > h:
            penalty += (ys[i] + rs[i] - h)**2
    
    # Penalty for overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                penalty += (min_dist - dist)**2
    
    # Objective: maximize sum of radii, so minimize negative sum plus penalty
    return -np.sum(rs) + 100.0 * penalty


def initial_guess(aspect_ratio, seed=0):
    """Create initial guess using a grid pattern."""
    rng = np.random.RandomState(seed)
    ar = aspect_ratio
    h = 2.0 / (1.0 + ar)
    w = 2.0 * ar / (1.0 + ar)
    
    # Grid layout
    cols = int(np.ceil(np.sqrt(N * ar)))
    rows = int(np.ceil(N / cols))
    
    xs = np.zeros(N)
    ys = np.zeros(N)
    rs = np.zeros(N)
    
    for i in range(N):
        col = i % cols
        row = i // cols
        xs[i] = (col + 0.5) * w / cols
        ys[i] = (row + 0.5) * h / rows
        rs[i] = 0.5 * min(w / cols, h / rows) * 0.95
    
    # Add small random perturbation
    xs += rng.uniform(-0.01, 0.01, N)
    ys += rng.uniform(-0.01, 0.01, N)
    rs += rng.uniform(-0.005, 0.005, N)
    rs = np.maximum(rs, 0.01)
    
    params = np.concatenate([xs, ys, rs])
    return params


def polish(circles, max_iter=200):
    """Iteratively adjust circles to reduce overlaps and improve packing."""
    circles = circles.copy()
    n = len(circles)
    
    for _ in range(max_iter):
        if elapsed() > DEADLINE - 5:
            break
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = circles[i, 2] + circles[j, 2]
                if dist < min_dist:
                    # Push apart
                    if dist < 1e-10:
                        dx, dy = 1e-10, 0
                        dist = 1e-10
                    overlap = (min_dist - dist) * 0.5
                    ux, uy = dx/dist, dy/dist
                    circles[i, 0] += ux * overlap
                    circles[i, 1] += uy * overlap
                    circles[j, 0] -= ux * overlap
                    circles[j, 1] -= uy * overlap
                    moved = True
        if not moved:
            break
    
    # Clamp to bounding box
    xs = circles[:, 0]
    ys = circles[:, 1]
    rs = circles[:, 2]
    x_min = np.min(xs - rs)
    x_max = np.max(xs + rs)
    y_min = np.min(ys - rs)
    y_max = np.max(ys + rs)
    width = x_max - x_min
    height = y_max - y_min
    perimeter = 2 * (width + height)
    if perimeter > 4.0:
        scale = 4.0 / perimeter
        # Scale positions and radii
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        circles[:, 0] = cx + (circles[:, 0] - cx) * scale
        circles[:, 1] = cy + (circles[:, 1] - cy) * scale
        circles[:, 2] *= scale
    
    return circles


def solve():
    best_circles = None
    best_score = -np.inf
    
    # Try multiple aspect ratios
    aspect_ratios = [0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0]
    
    for ar in aspect_ratios:
        if elapsed() > DEADLINE - 10:
            break
        
        # Try multiple seeds for diversity
        for seed in range(3):
            if elapsed() > DEADLINE - 5:
                break
            
            x0 = initial_guess(ar, seed=seed)
            
            # Use L-BFGS-B
            try:
                result = minimize(
                    objective,
                    x0,
                    args=(ar,),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-10}
                )
                
                if result.success or result.fun < objective(x0, ar):
                    params = result.x
                    n = N
                    xs = params[0:n]
                    ys = params[n:2*n]
                    rs = params[2*n:3*n]
                    
                    circles = np.column_stack([xs, ys, rs])
                    
                    # Polish
                    circles = polish(circles)
                    
                    # Check validity and compute score
                    if is_valid(circles):
                        score = np.sum(circles[:, 2])
                        if score > best_score:
                            best_score = score
                            best_circles = circles.copy()
            except Exception:
                pass
    
    # Fallback: simple grid if optimization failed
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
        # Scale to perimeter 4
        xs = best_circles[:, 0]
        ys = best_circles[:, 1]
        rs = best_circles[:, 2]
        x_min = np.min(xs - rs)
        x_max = np.max(xs + rs)
        y_min = np.min(ys - rs)
        y_max = np.max(ys + rs)
        width = x_max - x_min
        height = y_max - y_min
        perimeter = 2 * (width + height)
        if perimeter > 4.0:
            scale = 4.0 / perimeter
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            best_circles[:, 0] = cx + (best_circles[:, 0] - cx) * scale
            best_circles[:, 1] = cy + (best_circles[:, 1] - cy) * scale
            best_circles[:, 2] *= scale
    
    return best_circles


def construct():
    global START_TIME
    START_TIME = time.time()
    return solve()
