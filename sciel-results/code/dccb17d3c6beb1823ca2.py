"""Heilbronn triangles: maximize smallest triangle area for 11 points in equilateral triangle."""
import numpy as np
import time
import itertools

N = 11

# Equilateral triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, np.sqrt(3) / 2.0])
TRI_AREA = np.sqrt(3) / 4.0  # area of the equilateral triangle

# Precompute all triangle index combinations
COMBOS = list(itertools.combinations(range(N), 3))
COMBOS_ARR = np.array(COMBOS, dtype=np.int32)
N_COMBOS = len(COMBOS_ARR)


def _area_from_points(pts):
    """Compute areas of all triangles formed by points. Returns array of shape (N_COMBOS,)."""
    i, j, k = COMBOS_ARR[:, 0], COMBOS_ARR[:, 1], COMBOS_ARR[:, 2]
    p1 = pts[i]
    p2 = pts[j]
    p3 = pts[k]
    # Cross product magnitude / 2 = area
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    return 0.5 * np.abs(cross)


def _soft_min(areas, beta):
    """Soft minimum via log-sum-exp (negative log-sum-exp for min)."""
    # soft_min = -1/beta * log(sum(exp(-beta * areas)))
    # For stability: subtract max
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    return -max_val / beta - np.log(np.sum(np.exp(neg_beta_areas - max_val))) / beta


def _soft_min_grad(areas, beta):
    """Gradient of soft_min w.r.t. areas."""
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    exp_vals = np.exp(neg_beta_areas - max_val)
    weights = exp_vals / np.sum(exp_vals)
    return -weights


def _is_inside_triangle(pts):
    """Check if all points are inside the equilateral triangle (with small margin)."""
    eps = 1e-9
    x, y = pts[:, 0], pts[:, 1]
    # y >= 0
    if np.any(y < -eps):
        return False
    # y <= sqrt(3) * x  (left edge from A to C)
    if np.any(y > np.sqrt(3) * x + eps):
        return False
    # y <= -sqrt(3) * (x - 1)  (right edge from B to C)
    if np.any(y > -np.sqrt(3) * (x - 1) + eps):
        return False
    return True


def _project_to_triangle(pts):
    """Project points to be inside the triangle by clipping."""
    pts = pts.copy()
    x, y = pts[:, 0], pts[:, 1]
    # Clip y >= 0
    y = np.maximum(y, 0.0)
    # Clip y <= sqrt(3) * x
    y = np.minimum(y, np.sqrt(3) * x)
    # Clip y <= -sqrt(3) * (x - 1)
    y = np.minimum(y, -np.sqrt(3) * (x - 1))
    pts[:, 0] = x
    pts[:, 1] = y
    return pts


def _objective_and_grad(pts, beta):
    """Compute soft-min objective and its gradient w.r.t. point coordinates."""
    areas = _area_from_points(pts)
    obj = _soft_min(areas, beta)
    
    # Gradient: d(obj)/d(pts) = sum over combos of d(obj)/d(area) * d(area)/d(pts)
    w = _soft_min_grad(areas, beta)  # shape (N_COMBOS,)
    
    i, j, k = COMBOS_ARR[:, 0], COMBOS_ARR[:, 1], COMBOS_ARR[:, 2]
    p1 = pts[i]
    p2 = pts[j]
    p3 = pts[k]
    
    # area = 0.5 * |cross|, where cross = (p2-p1) x (p3-p1)
    # For gradient, we need d(area)/d(p1), d(area)/d(p2), d(area)/d(p3)
    # cross = (p2x-p1x)(p3y-p1y) - (p2y-p1y)(p3x-p1x)
    # d(cross)/d(p1x) = -(p3y-p1y) + (p2y-p1y) = p2y - p3y
    # d(cross)/d(p1y) = (p3x-p1x) - (p2x-p1x) = p2x - p3x
    # d(cross)/d(p2x) = (p3y-p1y)
    # d(cross)/d(p2y) = -(p3x-p1x)
    # d(cross)/d(p3x) = -(p2y-p1y)
    # d(cross)/d(p3y) = (p2x-p1x)
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    sign = np.sign(cross)  # since area = 0.5*|cross|, d(area)/d(cross) = 0.5*sign(cross)
    
    # d(area)/d(pts) for each point in each combo
    # Accumulate gradients
    grad = np.zeros_like(pts)
    
    # For p1: d(area)/d(p1x) = 0.5 * sign * (p2y - p3y), d(area)/d(p1y) = 0.5 * sign * (p3x - p2x)
    # For p2: d(area)/d(p2x) = 0.5 * sign * (p3y - p1y), d(area)/d(p2y) = 0.5 * sign * (-(p3x - p1x))
    # For p3: d(area)/d(p3x) = 0.5 * sign * (-(p2y - p1y)), d(area)/d(p3y) = 0.5 * sign * (p2x - p1x)
    
    w_half_sign = 0.5 * w * sign
    
    # Accumulate for p1
    np.add.at(grad, (i, 0), w_half_sign * (p2[:, 1] - p3[:, 1]))
    np.add.at(grad, (i, 1), w_half_sign * (p3[:, 0] - p2[:, 0]))
    
    # Accumulate for p2
    np.add.at(grad, (j, 0), w_half_sign * (p3[:, 1] - p1[:, 1]))
    np.add.at(grad, (j, 1), w_half_sign * (-(p3[:, 0] - p1[:, 0])))
    
    # Accumulate for p3
    np.add.at(grad, (k, 0), w_half_sign * (-(p2[:, 1] - p1[:, 1])))
    np.add.at(grad, (k, 1), w_half_sign * (p2[:, 0] - p1[:, 0]))
    
    # We want to MAXIMIZE the soft-min, so gradient of objective is -grad
    return obj, -grad


def _gradient_descent(pts0, betas, max_iter=5000, lr0=0.001, lr_decay=0.999, deadline=None):
    """Run gradient ascent on soft-min with multiple beta values."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_val = -np.inf
    
    for beta in betas:
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            
            obj, grad = _objective_and_grad(pts, beta)
            
            # Update
            lr = lr0 * (lr_decay ** it)
            pts = pts + lr * grad
            
            # Project to triangle
            pts = _project_to_triangle(pts)
            
            # Evaluate true min area
            areas = _area_from_points(pts)
            true_min = np.min(areas)
            if true_min > best_val:
                best_val = true_min
                best_pts = pts.copy()
        
        if deadline is not None and time.time() > deadline:
            break
    
    return best_pts, best_val


def _random_points(rng):
    """Generate random points inside the triangle."""
    pts = []
    for _ in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1.0:
                x = u + 0.5 * v
                y = (np.sqrt(3) / 2.0) * v
                pts.append([x, y])
                break
    return np.array(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.RandomState(42)
    
    # Multiple beta values for soft-min (larger beta = sharper min approximation)
    betas = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    best_pts = None
    best_score = 0.0
    
    # Multiple restarts
    n_restarts = 30
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Generate initial points
        pts0 = _random_points(rng)
        
        # Run gradient ascent
        pts_opt, score = _gradient_descent(
            pts0, betas, 
            max_iter=2000, 
            lr0=0.002, 
            lr_decay=0.998,
            deadline=deadline
        )
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()
        
        # Also try a simpler initialization: grid-like
        if restart % 5 == 0 and time.time() < deadline:
            # Try a more structured initial placement
            pts_grid = np.array([
                [0.5, np.sqrt(3)/6],       # center
                [0.25, np.sqrt(3)/12],     # 
                [0.75, np.sqrt(3)/12],
                [0.5, np.sqrt(3)/4],
                [0.15, 0.1],
                [0.85, 0.1],
                [0.3, np.sqrt(3)/5],
                [0.7, np.sqrt(3)/5],
                [0.5, np.sqrt(3)/8],
                [0.2, np.sqrt(3)/8],
                [0.8, np.sqrt(3)/8],
            ])
            # Perturb slightly
            pts_grid += rng.normal(0, 0.02, size=pts_grid.shape)
            pts_grid = _project_to_triangle(pts_grid)
            
            pts_opt, score = _gradient_descent(
                pts_grid, betas,
                max_iter=2000,
                lr0=0.002,
                lr_decay=0.998,
                deadline=deadline
            )
            
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
    
    # Final polish: run a few more iterations with high beta on the best point
    if best_pts is not None and time.time() < deadline:
        for beta in [100.0, 200.0, 500.0]:
            if time.time() > deadline:
                break
            pts_opt, score = _gradient_descent(
                best_pts, [beta],
                max_iter=3000,
                lr0=0.0005,
                lr_decay=0.9995,
                deadline=deadline
            )
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
    
    # Ensure points are strictly inside
    if best_pts is not None:
        best_pts = _project_to_triangle(best_pts)
        # Small inward shift to ensure strictly inside
        best_pts[:, 0] = np.clip(best_pts[:, 0], 1e-6, 1 - 1e-6)
        best_pts[:, 1] = np.clip(best_pts[:, 1], 1e-6, np.sqrt(3)/2 - 1e-6)
        # Re-project after clipping
        best_pts = _project_to_triangle(best_pts)
    
    if best_pts is None:
        # Fallback: random points
        best_pts = _random_points(rng)
    
    return best_pts
