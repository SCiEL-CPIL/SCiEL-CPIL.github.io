# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13
DEADLINE = 150.0
START_TIME = time.time()
SAFETY_MARGIN = 5.0

# Precompute triple indices for vectorized area computation
_ti, _tj, _tk = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _ti.append(i)
            _tj.append(j)
            _tk.append(k)
tri_i = np.array(_ti, dtype=np.int32)
tri_j = np.array(_tj, dtype=np.int32)
tri_k = np.array(_tk, dtype=np.int32)
n_triples = len(tri_i)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        h = ConvexHull(points)
        return h.volume
    except Exception:
        return 0.0

def _compute_areas(points):
    """Compute all triangle areas for the given points."""
    x = points[:, 0]
    y = points[:, 1]
    v1x = x[tri_j] - x[tri_i]
    v1y = y[tri_j] - y[tri_i]
    v2x = x[tri_k] - x[tri_i]
    v2y = y[tri_k] - y[tri_i]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _score(points):
    """Compute the score: min triangle area / hull area."""
    hull_area = _hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    areas = _compute_areas(points)
    min_area = np.min(areas)
    return float(min_area) / hull_area

def _soft_neg_score(points_flat, beta=50.0):
    """
    Objective for L-BFGS-B: Minimize -soft_min(areas, beta) / hull_area.
    """
    points = points_flat.reshape(-1, 2)
    hull_area = _hull_area(points)
    if hull_area < 1e-9:
        return 1e6  # Large penalty for degenerate hulls
    
    areas = _compute_areas(points)
    
    # Soft min calculation
    # To avoid overflow in exp, we can use the log-sum-exp trick, 
    # but for beta up to 400 and areas in reasonable range, direct exp is usually fine if shifted.
    # Let's shift by max area to be safe.
    max_a = np.max(areas)
    # soft_min = max_a - (1/beta) * log(sum(exp(-beta * (areas - max_a))))
    # We want to MINIMIZE -soft_min/hull_area
    # So we maximize soft_min/hull_area
    # Objective = - (soft_min / hull_area)
    
    if max_a < 1e-9:
        return 1e6
        
    exp_terms = np.exp(-beta * (areas - max_a))
    sum_exp = np.sum(exp_terms)
    if sum_exp == 0:
        return 1e6
        
    soft_min_val = max_a - (1.0 / beta) * np.log(sum_exp)
    
    return -(soft_min_val / hull_area)

def _hard_neg_score(points_flat):
    """
    Objective for NM/Powell: Minimize -score(points).
    """
    points = points_flat.reshape(-1, 2)
    s = _score(points)
    return -s

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0), phase=0.0):
    """Generate a regular n-gon."""
    if n == 1:
        return np.array([center])
    a = np.linspace(phase, phase + 2 * np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(a),
                            center[1] + radius * np.sin(a)])

def _generate_starts():
    """Generate diverse symmetric initial configurations."""
    starts = []
    
    # 1. Regular 13-gon
    starts.append(_regular_polygon(13, 1.0))
    
    # 2. 1 center + 12 outer
    outer = _regular_polygon(12, 1.0)
    center = np.array([[0.0, 0.0]])
    starts.append(np.vstack([outer, center]))
    
    # 3. 1 center + 6 mid + 6 outer
    outer = _regular_polygon(6, 1.0)
    mid = _regular_polygon(6, 0.5, phase=np.pi/6)
    center = np.array([[0.0, 0.0]])
    starts.append(np.vstack([outer, mid, center]))
    
    # 4. 1 center + 7 mid + 5 outer
    outer = _regular_polygon(5, 1.0)
    mid = _regular_polygon(7, 0.5, phase=np.pi/7)
    center = np.array([[0.0, 0.0]])
    starts.append(np.vstack([outer, mid, center]))
    
    # 5. 1 center + 5 mid + 7 outer
    outer = _regular_polygon(7, 1.0)
    mid = _regular_polygon(5, 0.5, phase=np.pi/5)
    center = np.array([[0.0, 0.0]])
    starts.append(np.vstack([outer, mid, center]))
    
    # 6. 1 center + 8 mid + 4 outer
    outer = _regular_polygon(4, 1.0)
    mid = _regular_polygon(8, 0.5, phase=np.pi/8)
    center = np.array([[0.0, 0.0]])
    starts.append(np.vstack([outer, mid, center]))
    
    # 7. 1 center + 4 mid + 8 outer
    outer = _regular_polygon(8, 1.0)
    mid = _regular_polygon(4, 0.5, phase=np.pi/4)
    center = np.array([[0.0, 0.0]])
    starts.append(np.vstack([outer, mid, center]))
    
    # 8. 1 center + 3 mid + 9 outer
    outer = _regular_polygon(9, 1.0)
    mid = _regular_polygon(3, 0.5, phase=np.pi/3)
    center = np.array([[0.0, 0.0]])
    starts.append(np.vstack([outer, mid, center]))
    
    # 9. 1 center + 9 mid + 3 outer
    outer = _regular_polygon(3, 1.0)
    mid = _regular_polygon(9, 0.5, phase=np.pi/9)
    center = np.array([[0.0, 0.0]])
    starts.append(np.vstack([outer, mid, center]))
    
    # 10. 1 center + 10 mid + 2 outer (2 outer is a line, but let's make it a segment or small polygon? 
    # Actually, 2 points don't form a polygon. Let's do 1 center + 10 mid + 2 outer as 2 points at 180 deg?
    # Better: 1 center + 10 mid + 2 outer is bad. Let's do 1 center + 10 mid + 2 outer as 2 points on a circle?
    # Let's stick to valid polygons. 
    # Try: 1 center + 11 outer?
    outer = _regular_polygon(11, 1.0)
    center = np.array([[0.0, 0.0]])
    starts.append(np.vstack([outer, center]))
    
    # 11. 1 center + 10 outer + 1 mid?
    outer = _regular_polygon(10, 1.0)
    mid = np.array([[0.5, 0.0]]) # Single mid point?
    center = np.array([[0.0, 0.0]])
    starts.append(np.vstack([outer, mid, center]))
    
    # 12. Perturbed Regular 13-gon
    base = _regular_polygon(13, 1.0)
    noise = np.random.normal(0, 0.05, (13, 2))
    starts.append(base + noise)
    
    # 13. Perturbed 1+12
    outer = _regular_polygon(12, 1.0)
    center = np.array([[0.0, 0.0]])
    base = np.vstack([outer, center])
    noise = np.random.normal(0, 0.05, (13, 2))
    starts.append(base + noise)
    
    return starts

def _refine(start_pts, deadline):
    """
    Refine a starting configuration using annealed soft-min L-BFGS-B, 
    then NM, then Powell.
    """
    current = start_pts.copy()
    best = start_pts.copy()
    best_score = _score(best)
    
    # Stage 1: Annealed Soft-Min L-BFGS-B
    betas = [8.0, 20.0, 50.0, 100.0, 200.0, 400.0]
    for beta in betas:
        if time.time() > deadline:
            break
        try:
            res = opt.minimize(
                lambda x: _soft_neg_score(x, beta),
                current.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 80}
            )
            cand = res.x.reshape(-1, 2)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best = cand.copy()
            current = cand
        except Exception:
            pass
            
    # Stage 2: Nelder-Mead Polish on Hard Objective
    if time.time() < deadline - 5:
        try:
            res = opt.minimize(
                _hard_neg_score,
                best.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'adaptive': True, 'xatol': 1e-11, 'fatol': 1e-13}
            )
            cand = res.x.reshape(-1, 2)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best = cand.copy()
        except Exception:
            pass
            
    # Stage 3: Powell Final Polish
    if time.time() < deadline - 2:
        try:
            res = opt.minimize(
                _hard_neg_score,
                best.flatten(),
                method='Powell',
                options={'maxiter': 300, 'xtol': 1e-10, 'ftol': 1e-12}
            )
            cand = res.x.reshape(-1, 2)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best = cand.copy()
        except Exception:
            pass
            
    return best, best_score

def construct():
    """
    Construct 13 points maximizing min triangle area / convex hull area.
    """
    start_time = time.time()
    deadline = START_TIME + DEADLINE - SAFETY_MARGIN
    
    starts = _generate_starts()
    
    best_points = None
    best_score = -float('inf')
    
    n_starts = len(starts)
    # Allocate time per start. 
    # We have ~145s budget. 13 starts. ~11s per start.
    # L-BFGS-B with 6 betas * 100 iters might be fast. NM and Powell might be slower.
    # Let's just loop and check time.
    
    for i, start_pts in enumerate(starts):
        if time.time() > deadline:
            break
            
        # Optional: Shuffle starts to ensure we don't just do the first few if time runs out?
        # The list is static. Let's just iterate.
        
        refined_pts, score = _refine(start_pts, deadline)
        
        if score > best_score:
            best_score = score
            best_points = refined_pts
            
    # Fallback if all failed or best_points is None
    if best_points is None or best_score <= 0:
        best_points = _regular_polygon(13, 1.0)
        
    return best_points
# EVOLVE-BLOCK-END
