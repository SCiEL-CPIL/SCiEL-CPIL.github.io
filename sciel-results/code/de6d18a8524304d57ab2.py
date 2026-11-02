"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0
DEADLINE = 145  # seconds, safety margin from 150s limit

# Vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

def _min_area_normalized(points):
    """Compute minimum triangle area normalized by the unit equilateral triangle area."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    # Compute all pairwise differences
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    
    # Cross product for all triples (i, j, k)
    cross = dx[:, :, None] * dy[:, None, :] - dy[:, :, None] * dx[:, None, :]
    areas = 0.5 * np.abs(cross)
    
    # Extract i < j < k
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return 0.0
    
    min_area = np.min(valid_areas)
    return min_area / TRI_AREA

def _is_inside_triangle(p):
    """Check if point p is inside the equilateral triangle."""
    x, y = p
    eps = 1e-9
    if y < -eps or y > S3H + eps:
        return False
    if x < -eps or x > 1 + eps:
        return False
    if y > S3H * (1 - x) + eps:
        return False
    if y > S3H * x + eps:
        return False
    return True

def _penalized_min_area(x):
    """Objective: minimize negative min area plus penalty for out-of-bounds."""
    points = x.reshape(-1, 2)
    
    # Penalty for points outside triangle
    penalty = 0.0
    for i in range(N):
        px, py = points[i]
        d_bottom = -py
        d_top = py - S3H
        d_left = -px
        d_right = px - 1
        d_slant_right = py - S3H * (1 - px)
        d_slant_left = py - S3H * px
        
        dist = max(0.0, d_bottom, d_top, d_left, d_right, d_slant_right, d_slant_left)
        penalty += dist ** 2
    
    min_a = _min_area_normalized(points)
    return -min_a + 10.0 * penalty

def _farthest_point_sampling(rng, n_points):
    """Generate n_points inside the triangle using Farthest Point Sampling."""
    points = []
    # Start with a random point near the center
    # Center of equilateral triangle is (0.5, S3H/3)
    start_x = 0.5 + (rng.random() - 0.5) * 0.1
    start_y = S3H / 3.0 + (rng.random() - 0.5) * 0.1
    # Ensure it's inside
    if not _is_inside_triangle([start_x, start_y]):
        start_x, start_y = 0.5, S3H / 3.0
    points.append([start_x, start_y])
    
    for _ in range(n_points - 1):
        # Find the point in the triangle that maximizes the minimum distance to existing points
        # We can approximate this by checking a grid or random candidates
        # For speed, we'll check a set of random candidates
        best_point = None
        best_min_dist = -1
        # Generate many candidates
        candidates = []
        for _ in range(500):
            # Random point in triangle
            u, v = rng.random(), rng.random()
            if u + v < 1:
                x = u + 0.5 * v
                y = S3H * v
                candidates.append([x, y])
        
        for c in candidates:
            # Check if inside
            if not _is_inside_triangle(c):
                continue
            # Calculate min distance to existing points
            min_dist = np.inf
            for p in points:
                d = np.linalg.norm(np.array(c) - np.array(p))
                if d < min_dist:
                    min_dist = d
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_point = c
                
        if best_point is not None:
            points.append(best_point)
        else:
            # Fallback to random if something went wrong
            while True:
                u, v = rng.random(), rng.random()
                if u + v < 1:
                    x = u + 0.5 * v
                    y = S3H * v
                    if _is_inside_triangle([x, y]):
                        points.append([x, y])
                        break
                        
    return np.array(points)

def construct():
    start_time = time.time()
    
    best_points = None
    best_score = -float('inf')
    rng = np.random.RandomState(42)
    
    # Phase 1: Multiple restarts with FPS init + Powell Optimization
    # Powell is robust for non-smooth functions.
    # We will run as many restarts as possible.
    
    restarts = 0
    while time.time() - start_time < DEADLINE - 15:
        restarts += 1
        
        # 1. Generate initial config using FPS
        init_pts = _farthest_point_sampling(rng, N)
        x0 = init_pts.flatten()
        
        # 2. Optimize with Powell
        # Powell doesn't have a strict maxiter that stops it quickly if it's slow, 
        # but it's usually fast for 22D.
        # We'll add a timeout check inside the callback if possible, or just trust it.
        # To be safe, we can limit maxiter.
        try:
            res = minimize(
                _penalized_min_area,
                x0,
                method='Powell',
                options={'maxiter': 200, 'maxfev': 2000}
            )
        except Exception:
            continue
            
        if time.time() - start_time > DEADLINE - 10:
            break
            
        pts = res.x.reshape(-1, 2)
        
        # Check validity
        valid = all(_is_inside_triangle(p) for p in pts)
        if valid:
            score = _min_area_normalized(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
        else:
            # If invalid, try to project back or just skip
            # We can try a quick repair: clamp to triangle
            repaired_pts = []
            for p in pts:
                x, y = p
                # Simple clamp logic
                # This is tricky, so we might just skip invalid results
                # But Powell with penalty should keep them close.
                pass
            
    # Fallback if no valid points found
    if best_points is None:
        best_points = _farthest_point_sampling(rng, N)
        best_score = _min_area_normalized(best_points)
        
    # Phase 2: Final Polish with Nelder-Mead
    # NM is good for final fine-tuning
    if time.time() - start_time < DEADLINE - 2:
        try:
            x0 = best_points.flatten()
            res_nm = minimize(
                _penalized_min_area,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            polished = res_nm.x.reshape(-1, 2)
            if all(_is_inside_triangle(p) for p in polished):
                score = _min_area_normalized(polished)
                if score > best_score:
                    best_points = polished
        except Exception:
            pass
            
    return best_points
# EVOLVE-BLOCK-END
