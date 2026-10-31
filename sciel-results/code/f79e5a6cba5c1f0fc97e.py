import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_indices = np.array(tri_indices)
I = tri_indices[:, 0]
J = tri_indices[:, 1]
K = tri_indices[:, 2]

def _min_area(points):
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    v1 = Pj - Pi
    v2 = Pk - Pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _penalty(points):
    x = points[:, 0]
    y = points[:, 1]
    v1 = np.maximum(0, -y)
    v2 = np.maximum(0, y - np.sqrt(3) * x)
    v3 = np.maximum(0, y - np.sqrt(3) * (1 - x))
    return np.sum(v1**2 + v2**2 + v3**2)

def _is_valid(points):
    return _penalty(points) < 1e-6

def _project_point(p):
    x, y = p
    # If y < 0
    if y < 0:
        y = 0
    # If y > sqrt(3) * x
    if y > np.sqrt(3) * x:
        dot = x * 0.5 + y * (np.sqrt(3)/2)
        x = dot * 0.5
        y = dot * (np.sqrt(3)/2)
    # If y > sqrt(3) * (1 - x)
    if y > np.sqrt(3) * (1 - x):
        vx = x - 1
        vy = y
        dot = vx * 0.5 + vy * (-np.sqrt(3)/2)
        x = 1 + dot * 0.5
        y = 0 + dot * (-np.sqrt(3)/2)
        
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, H)
    return np.array([x, y])

def _soft_objective(flat_points, beta=50.0):
    """
    Minimizes a smooth approximation of the negative min area.
    Lower is better.
    """
    points = flat_points.reshape(-1, 2)
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    v1 = Pj - Pi
    v2 = Pk - Pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Add penalty for being outside
    pen = _penalty(points)
    if pen > 1e-6:
        return 1000.0 + pen
        
    # Soft min: -log(sum(exp(-beta * areas))) / beta
    vals = -beta * areas
    max_val = np.max(vals)
    log_sum_exp = max_val + np.log(np.sum(np.exp(vals - max_val)))
    soft_min_val = -log_sum_exp / beta
    
    # We want to minimize -soft_min_val
    return -soft_min_val

def _hard_objective(flat_points):
    """
    Direct minimization of negative min area.
    """
    points = flat_points.reshape(-1, 2)
    if not _is_valid(points):
        return 1000.0 + _penalty(points)
    return -_min_area(points)

def _generate_lattice():
    """
    Generate a triangular lattice configuration.
    """
    s = 0.21
    grid_pts = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = i * s
            y = j * s * np.sqrt(3)
            if i % 2 != 0:
                y += s * np.sqrt(3) / 2
            grid_pts.append([x, y])
    grid_pts = np.array(grid_pts)
    
    cx, cy = 0.5, H/3
    grid_pts += [cx, cy]
    
    mask = _penalty(grid_pts) == 0
    inside_pts = grid_pts[mask]
    
    if len(inside_pts) >= N:
        idx = np.random.choice(len(inside_pts), N, replace=False)
        return inside_pts[idx].copy()
    else:
        pts_list = list(inside_pts)
        while len(pts_list) < N:
            u = np.random.random()
            v = np.random.random()
            if u + v < 1:
                x = u + 0.5 * v
                y = H * v
                pts_list.append([x, y])
        return np.array(pts_list)

def _random_init(rng):
    pts = []
    for _ in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v < 1:
                x = u + 0.5 * v
                y = H * v
                pts.append([x, y])
                break
    return np.array(pts)

def _coordinate_descent(points, max_iter=200, tol=1e-6):
    """
    Coordinate descent to refine points.
    """
    pts = points.copy()
    best_score = _min_area(pts)
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            # Try moving point i in small steps
            best_local = best_score
            best_pt = pts[i].copy()
            
            # Adaptive step sizes
            for step in [0.05, 0.02, 0.01, 0.005, 0.001]:
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        new_pt = pts[i] + [dx, dy]
                        new_pt = _project_point(new_pt)
                        
                        pts[i] = new_pt
                        score = _min_area(pts)
                        
                        if score > best_local:
                            best_local = score
                            best_pt = new_pt.copy()
                            improved = True
                        else:
                            pts[i] = best_pt
            
            pts[i] = best_pt
            best_score = _min_area(pts)
        
        if not improved:
            break
            
    return pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer for 150s limit
    
    rng = np.random.default_rng(42)
    
    # Initial candidates
    candidates = []
    # 1. Lattice
    candidates.append(_generate_lattice().flatten())
    # 2. Random
    for _ in range(5):
        pts = _random_init(rng)
        candidates.append(pts.flatten())
    
    best_score = -np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if _is_valid(pts):
            score = _min_area(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
                
    # Evaluate initial candidates
    for cand in candidates:
        pts = cand.reshape(-1, 2)
        update_best(pts)
        
    # Strategy 1: Simulated Annealing
    T_start = 0.1
    T_end = 1e-5
    alpha = 0.9999
    
    current_pts = best_points.copy()
    current_score = best_score
    T = T_start
    
    while time.time() < deadline - 30:  # Leave 30s for local refinement
        # Pick a random point to move
        i = rng.integers(0, N)
        
        # Perturb
        step = 0.05 * (T / T_start + 0.1)
        dx = rng.normal(0, step)
        dy = rng.normal(0, step)
        
        new_pts = current_pts.copy()
        new_pts[i] = new_pts[i] + [dx, dy]
        new_pts[i] = _project_point(new_pts[i])
        
        new_score = _min_area(new_pts)
        
        # Accept or reject
        if new_score > current_score or rng.random() < np.exp((new_score - current_score) / T):
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_pts.copy()
        
        T *= alpha
    
    # Strategy 2: Local refinement with Nelder-Mead on the best point
    if best_points is not None and time.time() < deadline - 10:
        flat_best = best_points.flatten()
        try:
            res = minimize(
                _hard_objective,
                flat_best,
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 2000, 'maxfev': 5000}
            )
            res_pts = res.x.reshape(-1, 2)
            if _is_valid(res_pts):
                res_score = _min_area(res_pts)
                if res_score > best_score:
                    best_score = res_score
                    best_points = res_pts.copy()
        except Exception:
            pass
    
    # Strategy 3: Coordinate Descent for final precision
    if best_points is not None and time.time() < deadline - 5:
        cd_pts, cd_score = _coordinate_descent(best_points, max_iter=100)
        if cd_score > best_score:
            best_score = cd_score
            best_points = cd_pts.copy()
    
    # Ensure we return a valid result
    if best_points is None:
        best_points = _random_init(rng)
    
    return best_points

