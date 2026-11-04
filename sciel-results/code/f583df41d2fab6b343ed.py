"""Optimized program for Heilbronn convex regions n=14."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random
from itertools import combinations

def _triangle_area_vec(pts):
    """Compute all triangle areas for a set of points (n,2). Returns array of size C(n,3)."""
    n = len(pts)
    areas = np.empty((n*(n-1)*(n-2))//6)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Cross product magnitude / 2
                ax = pts[j,0] - pts[i,0]
                ay = pts[j,1] - pts[i,1]
                bx = pts[k,0] - pts[i,0]
                by = pts[k,1] - pts[i,1]
                areas[idx] = 0.5 * abs(ax*by - ay*bx)
                idx += 1
    return areas

def _min_triangle_area(pts):
    """Minimum triangle area over all triples."""
    return np.min(_triangle_area_vec(pts))

def _convex_hull_area(pts):
    """Area of convex hull of points."""
    # Sort by x, then y
    pts_sorted = pts[np.lexsort((pts[:,1], pts[:,0]))]
    n = len(pts_sorted)
    if n < 3:
        return 0.0
    # Build lower hull
    lower = []
    for p in pts_sorted:
        while len(lower) >= 2:
            p1 = np.array(lower[-1])
            p2 = np.array(p)
            p0 = np.array(lower[-2])
            cross = (p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0])
            if cross <= 1e-12:
                lower.pop()
            else:
                break
        lower.append(p)
    # Build upper hull
    upper = []
    for p in reversed(pts_sorted):
        while len(upper) >= 2:
            p1 = np.array(upper[-1])
            p2 = np.array(p)
            p0 = np.array(upper[-2])
            cross = (p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0])
            if cross <= 1e-12:
                upper.pop()
            else:
                break
        upper.append(p)
    # Combine
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    # Shoelace formula
    x = hull[:,0]
    y = hull[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area

def _normalize(pts):
    """Normalize points so convex hull has unit area, centered at origin."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return pts
    scale = 1.0 / math.sqrt(hull_area)
    centroid = np.mean(pts, axis=0)
    return (pts - centroid) * scale

def _min_tri_area_normalized(pts):
    """Min triangle area after normalizing to unit hull area."""
    pts_norm = _normalize(pts)
    return _min_triangle_area(pts_norm)

def _initial_configurations():
    """Generate several structured initial configurations."""
    configs = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*math.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    configs.append(pts.copy())
    
    # 2. Two concentric heptagons (7 outer, 7 inner)
    angles_outer = np.linspace(0, 2*math.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2*math.pi, 7, endpoint=False) + math.pi/7
    outer = np.column_stack([1.0*np.cos(angles_outer), 1.0*np.sin(angles_outer)])
    inner = np.column_stack([0.5*np.cos(angles_inner), 0.5*np.sin(angles_inner)])
    pts = np.vstack([outer, inner])
    configs.append(pts.copy())
    
    # 3. Grid-like: 4x4 grid minus 2 corners, adjusted
    # Actually let's do a 3x5 grid plus 1 extra, or similar
    # 4x4 grid = 16, we need 14. Remove 2.
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    grid_pts = []
    for x in xs:
        for y in ys:
            grid_pts.append([x, y])
    # Remove 2 random-ish corners to get 14
    grid_pts = grid_pts[:14]
    pts = np.array(grid_pts)
    configs.append(pts.copy())
    
    # 4. Regular 7-gon + 7 points at radius 0.45, offset
    angles_outer = np.linspace(0, 2*math.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2*math.pi, 7, endpoint=False) + math.pi/14
    outer = np.column_stack([1.0*np.cos(angles_outer), 1.0*np.sin(angles_outer)])
    inner = np.column_stack([0.45*np.cos(angles_inner), 0.45*np.sin(angles_inner)])
    pts = np.vstack([outer, inner])
    configs.append(pts.copy())
    
    # 5. 14 points on a circle but with slight radial perturbation
    angles = np.linspace(0, 2*math.pi, N, endpoint=False)
    radii = np.ones(N)
    # Perturb some radii
    for i in range(0, N, 2):
        radii[i] = 0.9
    pts = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])
    configs.append(pts.copy())
    
    # 6. Three concentric rings: 5 outer, 5 middle, 4 inner
    a1 = np.linspace(0, 2*math.pi, 5, endpoint=False)
    a2 = np.linspace(0, 2*math.pi, 5, endpoint=False) + math.pi/5
    a3 = np.linspace(0, 2*math.pi, 4, endpoint=False) + math.pi/8
    r1 = np.column_stack([1.0*np.cos(a1), 1.0*np.sin(a1)])
    r2 = np.column_stack([0.6*np.cos(a2), 0.6*np.sin(a2)])
    r3 = np.column_stack([0.25*np.cos(a3), 0.25*np.sin(a3)])
    pts = np.vstack([r1, r2, r3])
    configs.append(pts.copy())
    
    # 7. Slightly irregular polygon
    angles = np.linspace(0, 2*math.pi, N, endpoint=False)
    rng = random.Random(42)
    radii = np.array([1.0 + 0.1*rng.gauss() for _ in range(N)])
    pts = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])
    configs.append(pts.copy())
    
    # 8. Hexagonal lattice fragment
    # Generate hex grid points
    hex_pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i + 0.5*(j % 2)
            y = j * math.sqrt(3)/2
            if x*x + y*y <= 4.0:
                hex_pts.append([x, y])
    # Take first 14
    if len(hex_pts) >= 14:
        pts = np.array(hex_pts[:14])
        configs.append(pts.copy())
    
    return configs

def _local_optimize(pts, max_iter=500, step_size=0.05):
    """Simple coordinate descent / Nelder-Mead style local optimization."""
    best_pts = pts.copy()
    best_score = _score(best_pts)
    rng = random.Random(12345)
    
    # Use a simple hill-climbing with random perturbations
    current_score = best_score
    for iteration in range(max_iter):
        improved = False
        # Try random perturbations
        for _ in range(20):
            trial = best_pts.copy()
            # Perturb one or two points
            n_perturb = rng.choice([1, 2])
            indices = rng.sample(range(N), n_perturb)
            for idx in indices:
                dx = rng.gauss(0, step_size)
                dy = rng.gauss(0, step_size)
                trial[idx, 0] += dx
                trial[idx, 1] += dy
            trial_score = _score(trial)
            if trial_score > current_score + 1e-10:
                best_pts = trial
                current_score = trial_score
                improved = True
                break
        if not improved:
            # Reduce step size
            step_size *= 0.8
            if step_size < 1e-6:
                break
    
    return best_pts, best_score

def construct():
    """Construct 14 points to maximize min triangle area / convex hull area."""
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_pts = None
    best_score = 0.0
    
    # Generate initial configurations
    configs = _initial_configurations()
    
    # Evaluate and optimize each configuration
    for i, config in enumerate(configs):
        if time.time() > deadline:
            break
        
        # Normalize
        config_norm = _normalize(config)
        config_score = _score(config_norm)
        
        if config_score > best_score:
            best_pts = config_norm.copy()
            best_score = config_score
        
        # Local optimization
        if time.time() < deadline - 10:
            opt_pts, opt_score = _local_optimize(config_norm, max_iter=300, step_size=0.03)
            if opt_score > best_score:
                best_pts = opt_pts
                best_score = opt_score
    
    # If we have time, do more restarts with random perturbations of best
    if best_pts is not None and time.time() < deadline - 5:
        rng = random.Random(999)
        current_best = best_pts.copy()
        current_score = best_score
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            trial = current_best.copy()
            # Small random perturbation
            perturb_idx = rng.sample(range(N), rng.choice([1,2,3]))
            for idx in perturb_idx:
                trial[idx, 0] += rng.gauss(0, 0.02)
                trial[idx, 1] += rng.gauss(0, 0.02)
            trial_score = _score(trial)
            if trial_score > current_score + 1e-10:
                current_best = trial
                current_score = trial_score
                best_pts = trial
                best_score = trial_score
            # Also try local optimization from this point
            if time.time() < deadline - 5:
                opt_pts, opt_score = _local_optimize(trial, max_iter=200, step_size=0.02)
                if opt_score > best_score:
                    best_pts = opt_pts
                    best_score = opt_score
                    current_best = best_pts
                    current_score = best_score
    
    # Final: ensure we return a valid point set
    if best_pts is None:
        # Fallback: regular 14-gon
        angles = np.linspace(0, 2*math.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts
# EVOLVE-BLOCK-END
