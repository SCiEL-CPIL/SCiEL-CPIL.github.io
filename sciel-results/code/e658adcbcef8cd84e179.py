import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.spatial import Delaunay
import itertools

N = 11
H = np.sqrt(3) / 2.0
# Triangle vertices: A=(0,0), B=(1,0), C=(0.5, H)

def _triangle_area(p1, p2, p3):
    """Area of triangle formed by 3 points (2D)."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

def _min_area(pts):
    """Smallest triangle area among all triples."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_a = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _triangle_area(pts[i], pts[j], pts[k])
                if a < min_a:
                    min_a = a
    return min_a

def _soft_min_area(pts, k=100.0):
    """Smooth approximation of min area using log-sum-exp on areas^2 (to keep positive)."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = _triangle_area(pts[i], pts[j], pts[k])
                areas.append(a)
    areas = np.array(areas)
    # Use log-sum-exp of (a/k)^p for large p to approximate min
    # Actually, to approximate min, we use -log-sum-exp(-a/k) / k approximately
    # Better: use a = areas, compute m = min, then softmin = -1/k * log(sum(exp(-k*a)))
    # This is expensive; use a simpler approach: penalize small areas
    # Let's use: objective = -softmin where softmin approximates min
    # For efficiency, use: -1/k * log(sum(exp(-k*a_i)))
    # But this is sensitive. Let's use a different smooth min:
    # f = (sum(a_i^p))^(1/p) for p -> -infty gives min. Use p=-10 or so.
    # Actually, let's just use the raw min for speed and use gradient-based on a smoothed version.
    # For now, return the actual min for evaluation, but use a different objective for optimization.
    return np.min(areas)

def _objective_smooth(pts_flat, k=50.0):
    """Objective to maximize: smooth approximation of min triangle area.
    We minimize the negative of the soft-min.
    pts_flat is a 2D array reshaped to 22 elements.
    """
    pts = pts_flat.reshape(-1, 2)
    n = len(pts)
    if n < 3:
        return 0.0
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k_idx in range(j+1, n):
                a = _triangle_area(pts[i], pts[j], pts[k_idx])
                areas.append(a)
    areas = np.array(areas)
    # Soft min: -1/k * log(sum(exp(-k*a)))
    # To avoid overflow, shift by min
    a_min = np.min(areas)
    exp_terms = np.exp(-k * (areas - a_min))
    softmin = a_min - np.log(np.sum(exp_terms)) / k
    return -softmin

def _constraint_check(pts):
    """Check if all points are inside the triangle."""
    for p in pts:
        x, y = p
        if y < -1e-9 or y > H + 1e-9:
            return False
        # Line AB: y >= 0 (already checked)
        # Line AC: y <= sqrt(3)*x -> x >= y/sqrt(3)
        if x < y / np.sqrt(3) - 1e-9:
            return False
        # Line BC: y <= -sqrt(3)*(x - 1) -> y <= sqrt(3)*(1-x)
        if y > np.sqrt(3) * (1 - x) + 1e-9:
            return False
    return True

def _penalty(pts, weight=1000.0):
    """Penalty for points outside the triangle."""
    penalty = 0.0
    for p in pts:
        x, y = p
        # Distance from edges
        # Edge AB: y=0, inside is y>=0
        if y < 0:
            penalty += weight * (-y)
        # Edge AC: y = sqrt(3)*x, inside is y <= sqrt(3)*x i.e., sqrt(3)*x - y >= 0
        d1 = np.sqrt(3) * x - y
        if d1 < 0:
            penalty += weight * (-d1)
        # Edge BC: y = -sqrt(3)*(x-1), inside is y <= -sqrt(3)*(x-1) i.e., -sqrt(3)*x + sqrt(3) - y >= 0
        d2 = -np.sqrt(3) * x + np.sqrt(3) - y
        if d2 < 0:
            penalty += weight * (-d2)
    return penalty

def _objective_with_penalty(pts_flat, k=50.0, weight=1000.0):
    pts = pts_flat.reshape(-1, 2)
    base = _objective_smooth(pts, k)
    pen = _penalty(pts, weight)
    return base + pen

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    best_pts = None
    best_score = -float('inf')
    
    # Initial guess: regular grid + jitter
    # Use a 3x3 grid inside the triangle plus 2 more points
    # Or use a known good configuration: place points on a hexagonal lattice clipped to triangle
    
    # Start with a simple grid
    def _initial_grid():
        pts = []
        # 4x4 grid in barycentric coordinates
        for i in range(4):
            for j in range(4 - i):
                for k in range(4 - i - j):
                    if i + j + k == 3:  # Only points on the "level" 3
                        # Barycentric: (i/3, j/3, k/3)
                        u = i / 3.0
                        v = j / 3.0
                        w = k / 3.0
                        x = u * 0.0 + v * 1.0 + w * 0.5
                        y = u * 0.0 + v * 0.0 + w * H
                        pts.append([x, y])
        # This gives 6 points. Need 11.
        # Let's use a different approach: 11 points in a roughly hexagonal pattern
        return None
    
    # Use a better initial guess: place points on a small hexagonal lattice
    def _hex_initial():
        pts = []
        # Hex lattice with spacing ~0.25
        s = 0.25
        for i in range(-2, 3):
            for j in range(-2, 3):
                x = i * s + (j % 2) * s * 0.5
                y = j * s * np.sqrt(3) / 2
                if _constraint_check(np.array([x, y])):
                    pts.append([x, y])
        # We need exactly 11 points. Take the 11 closest to center
        if len(pts) >= 11:
            # Sort by distance to centroid
            centroid = np.mean(pts, axis=0)
            dists = [np.linalg.norm(p - centroid) for p in pts]
            order = np.argsort(dists)
            return np.array(pts)[:11]
        # If not enough, add random
        while len(pts) < 11:
            x = np.random.uniform(0.1, 0.9)
            y = np.random.uniform(0.1, H * 0.8)
            if _constraint_check(np.array([x, y])):
                pts.append([x, y])
        return np.array(pts)
    
    rng = np.random.RandomState(42)
    
    # Generate multiple random starts
    starts = []
    
    # Start 1: Hex lattice
    starts.append(_hex_initial())
    
    # Start 2: Random with rejection sampling
    def _random_start(seed):
        rng_local = np.random.RandomState(seed)
        pts = []
        while len(pts) < 11:
            u = rng_local.random()
            v = rng_local.random()
            if u + v <= 1:
                x = u + 0.5 * v
                y = H * v
                # Add small margin from edges
                if x > 0.02 and y > 0.02 and x < 0.98 and y < H * 0.98:
                    pts.append([x, y])
        return np.array(pts)
    
    for s in range(5):
        starts.append(_random_start(s))
    
    # Evaluate initial starts
    for pts in starts:
        if not _constraint_check(pts):
            continue
        score = _min_area(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback: very simple configuration
        best_pts = _random_start(0)
        best_score = _min_area(best_pts)
    
    # Local optimization using scipy
    for start_idx in range(len(starts)):
        if time.time() > deadline - 5.0:
            break
        
        x0 = starts[start_idx].flatten()
        if not _constraint_check(starts[start_idx]):
            continue
        
        # Use L-BFGS-B with bounds
        # Bounds: each point must be inside triangle
        # We'll use penalty method
        
        options = {'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
        
        try:
            res = minimize(_objective_with_penalty, x0, method='L-BFGS-B', 
                          options=options, args=(50.0, 1000.0))
            
            if time.time() > deadline - 5.0:
                break
            
            opt_pts = res.x.reshape(-1, 2)
            if _constraint_check(opt_pts):
                opt_score = _min_area(opt_pts)
                if opt_score > best_score:
                    best_score = opt_score
                    best_pts = opt_pts.copy()
                    
        except Exception as e:
            pass
    
    # Final: try a few more random restarts with local opt if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        for s in range(10, 30):
            if time.time() > deadline - 10.0:
                break
            x0 = _random_start(s).flatten()
            if not _constraint_check(_random_start(s)):
                continue
            try:
                options = {'maxiter': 100, 'ftol': 1e-8}
                res = minimize(_objective_with_penalty, x0, method='L-BFGS-B',
                              options=options, args=(50.0, 1000.0))
                if time.time() > deadline - 5.0:
                    break
                opt_pts = res.x.reshape(-1, 2)
                if _constraint_check(opt_pts):
                    opt_score = _min_area(opt_pts)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_pts = opt_pts.copy()
            except:
                pass
    
    # Ensure we return exactly 11 points inside the triangle
    if best_pts is None or len(best_pts) != 11:
        best_pts = _random_start(0)
    
    return best_pts
