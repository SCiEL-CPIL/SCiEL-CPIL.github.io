"""Heilbronn triangles: place 11 points in unit equilateral triangle to maximize min triangle area."""
import numpy as np
import time
import itertools
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import Delaunay

N = 11
H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute all C(11,3) = 165 triangle index triplets
_idx_i, _idx_j, _idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _idx_i.append(i)
            _idx_j.append(j)
            _idx_k.append(k)
_idx_i = np.array(_idx_i)
_idx_j = np.array(_idx_j)
_idx_k = np.array(_idx_k)
_n_tris = len(_idx_i)

def _min_area_fast(points):
    """Compute minimum triangle area among all C(N,3) triples. points: (N,2)."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    return 0.5 * np.min(cross)

def _soft_min_area(points, beta):
    """Compute soft-min of triangle areas using log-sum-exp trick."""
    p1 = points[_idx_i]
    p2 = points[_idx_j]
    p3 = points[_idx_k]
    cross = np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                   (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))
    areas = 0.5 * cross
    
    # Log-sum-exp for stability: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, subtract max(areas) which is -min(-areas)
    # Actually, we want min(areas).
    # soft_min = -1/beta * log(sum(exp(-beta * a_i)))
    # Let m = min(a_i). exp(-beta * a_i) = exp(-beta * (a_i - m)) * exp(-beta * m)
    # sum(exp(-beta * a_i)) = exp(-beta * m) * sum(exp(-beta * (a_i - m)))
    # log(sum(...)) = -beta * m + log(sum(exp(-beta * (a_i - m))))
    # soft_min = m - 1/beta * log(sum(exp(-beta * (a_i - m))))
    
    m = np.min(areas)
    # If m is very small, areas - m is non-negative.
    # exp(-beta * (areas - m)) is <= 1.
    # Sum is between 1 and 165.
    # log(sum) is safe.
    
    # Handle case where beta is large and areas-m is large -> exp underflows to 0.
    # That's fine, it just means that term doesn't contribute.
    
    # We need to be careful if all areas are 0 (degenerate).
    if m == 0:
        return 0.0
        
    shifted = areas - m
    # exp(-beta * shifted)
    # If beta * shifted > 700, exp underflows. This is okay.
    exp_terms = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_terms))
    
    return m - (1.0 / beta) * log_sum

def _inside_penalty(points, margin=0.001):
    """Smooth penalty for points outside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    # Triangle constraints:
    # 1. y >= 0
    # 2. y <= sqrt(3) * x  => sqrt(3)*x - y >= 0
    # 3. y <= sqrt(3) * (1-x) => sqrt(3)*(1-x) - y >= 0
    
    viol1 = np.maximum(0.0, -y + margin)
    viol2 = np.maximum(0.0, y - np.sqrt(3) * x + margin)
    viol3 = np.maximum(0.0, y - np.sqrt(3) * (1.0 - x) + margin)
    return np.sum(viol1**2) + np.sum(viol2**2) + np.sum(viol3**2)

def _objective_hard(params):
    """Objective for hard min area. Returns negative min area + penalty."""
    points = params.reshape(N, 2)
    pen = _inside_penalty(points)
    if pen > 1e-4:
        return -_min_area_fast(points) + 1000.0 * pen
    return -_min_area_fast(points)

def _objective_soft(params, beta):
    """Objective for soft min area. Returns negative soft min area + penalty."""
    points = params.reshape(N, 2)
    pen = _inside_penalty(points)
    if pen > 1e-4:
        return -_soft_min_area(points, beta) + 1000.0 * pen
    return -_soft_min_area(points, beta)

def _init_symmetric():
    """Generate a good symmetric starting configuration."""
    points = np.zeros((N, 2))
    # 4-3-3-1 pattern
    row_ys = [0.08 * H, 0.30 * H, 0.55 * H, 0.78 * H]
    row_ns = [4, 3, 3, 1]
    
    pi = 0
    for y, n_pts in zip(row_ys, row_ns):
        x_left = 0.5 * y / H
        x_right = 1.0 - 0.5 * y / H
        width = x_right - x_left
        
        if n_pts == 1:
            points[pi, 0] = 0.5
            points[pi, 1] = y
            pi += 1
        else:
            margin_frac = 0.15
            usable = width * (1 - 2 * margin_frac)
            start = x_left + width * margin_frac
            if n_pts > 1:
                xs = start + np.linspace(0, usable, n_pts)
            else:
                xs = np.array([0.5])
            for x in xs:
                points[pi, 0] = x
                points[pi, 1] = y
                pi += 1
    return points

def _init_random(rng):
    """Generate random points inside the triangle."""
    points = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1:
                points[i, 0] = v + 0.5 * (1 - u - v)
                points[i, 1] = H * (1 - u - v)
                break
    return points

def _polish_soft_min(points, time_limit):
    """Polish points using soft-min annealing."""
    start = time.time()
    current_points = points.copy()
    best_score = _min_area_fast(current_points)
    
    # Annealing schedule for beta
    betas = [10.0, 50.0, 100.0, 500.0, 1000.0]
    
    for beta in betas:
        if time.time() > time_limit:
            break
            
        def obj_soft(params):
            return _objective_soft(params, beta)
        
        try:
            res = minimize(obj_soft, current_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            
            new_points = res.x.reshape(N, 2)
            
            # Check validity
            x, y = new_points[:, 0], new_points[:, 1]
            if not (np.all(y >= -1e-8) and np.all(y <= np.sqrt(3)*x + 1e-8) and np.all(y <= np.sqrt(3)*(1-x) + 1e-8)):
                # Clamp
                new_points = _clamp_to_triangle(new_points)
            
            new_score = _min_area_fast(new_points)
            if new_score > best_score:
                best_score = new_score
                current_points = new_points.copy()
                
        except Exception:
            pass
            
    return current_points, best_score

def _clamp_to_triangle(points):
    """Clamp points to be inside the triangle."""
    points = points.copy()
    for i in range(N):
        x, y = points[i]
        y = max(0.0, min(H, y))
        x_min = y / np.sqrt(3)
        x_max = 1.0 - y / np.sqrt(3)
        if x_min > x_max:
            x, y = 0.5, 0.0
        else:
            x = max(x_min, min(x_max, x))
        points[i] = [x, y]
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 145 # Safety margin
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Initial candidates
    candidates = []
    init_pts = _init_symmetric()
    candidates.append(init_pts)
    
    # Add a few random inits
    for _ in range(3):
        candidates.append(_init_random(rng))
        
    best_points = None
    best_score = -np.inf
    
    # Evaluate initial candidates
    for pts in candidates:
        pts = _clamp_to_triangle(pts)
        score = _min_area_fast(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
            
    # Phase 2: Global Search with Differential Evolution
    # Use hard min objective for DE as it is more robust for global search
    bounds = [(0, 1), (0, H)] * N
    
    # Run DE for ~50 seconds
    de_deadline = start_time + 50
    n_de = 0
    while time.time() < de_deadline:
        n_de += 1
        pop_size = 20
        max_iter = 30
        
        try:
            result = differential_evolution(
                _objective_hard,
                bounds,
                maxiter=max_iter,
                popsize=pop_size,
                tol=1e-6,
                seed=n_de * 137,
                polish=False,
                init='latinhypercube',
                mutation=(0.5, 0.9),
                recombination=0.8
            )
        except Exception:
            continue
        
        de_points = result.x.reshape(N, 2)
        de_points = _clamp_to_triangle(de_points)
        de_score = _min_area_fast(de_points)
        
        if de_score > best_score:
            best_score = de_score
            best_points = de_points.copy()
            
        # Keep top DE results for polishing
        # We'll just polish the best one found so far in the next phase
        # But let's store a few good ones to try
        if de_score > best_score - 0.5: # Within 0.5 of best
             candidates.append(de_points)
             
    # Phase 3: Local Polish with Soft-Min Annealing
    # We will polish the best point and a few other good candidates
    # Time limit for polishing: remaining time - 10s for final check
    polish_deadline = deadline - 10
    
    # Ensure best_points is in candidates
    if best_points is not None:
        # Find index of best or append
        found = False
        for i, c in enumerate(candidates):
             if np.allclose(c, best_points):
                  found = True
                  break
        if not found:
             candidates.append(best_points.copy())
    
    # Sort candidates by score
    scored_candidates = []
    for pts in candidates:
        s = _min_area_fast(pts)
        scored_candidates.append((s, pts))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 3 unique-ish candidates
    unique_candidates = []
    for s, pts in scored_candidates:
        if len(unique_candidates) >= 3:
            break
        # Check if too similar to existing
        is_unique = True
        for u_pts in unique_candidates:
            if np.linalg.norm(pts - u_pts) < 0.05:
                is_unique = False
                break
        if is_unique:
            unique_candidates.append(pts)
    
    for pts in unique_candidates:
        if time.time() > polish_deadline:
            break
        polished_pts, polished_score = _polish_soft_min(pts, polish_deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts
    
    # Phase 4: Final High-Precision Polish on Hard Objective
    if time.time() < deadline - 2 and best_points is not None:
        try:
            res = minimize(_objective_hard, best_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
            final_pts = res.x.reshape(N, 2)
            final_pts = _clamp_to_triangle(final_pts)
            final_score = _min_area_fast(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_points = final_pts
        except Exception:
            pass
            
    if best_points is None:
        best_points = _init_symmetric()
        
    return _clamp_to_triangle(best_points)
