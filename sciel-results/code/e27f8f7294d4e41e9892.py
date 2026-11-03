"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from itertools import combinations

DEADLINE = 135.0  # Safety margin before 150s kill


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _all_triangle_areas(pts):
    """Compute areas of all C(n,3) triangles."""
    n = len(pts)
    areas = np.empty((n*(n-1)*(n-2))//6)
    idx = 0
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx-ax)*(cy-ay) - (cx-ax)*(by-ay))
                areas[idx] = area
                idx += 1
    return areas


def _score(pts):
    """Compute min triangle area / hull area."""
    hull_a = _hull_area(pts)
    if hull_a <= 0:
        return 0.0
    areas = _all_triangle_areas(pts)
    return np.min(areas) / hull_a


def _normalize_unit_area(pts):
    """Scale points so convex hull area is 1."""
    hull_a = _hull_area(pts)
    if hull_a > 0:
        return pts / np.sqrt(hull_a)
    return pts


def _initial_regular_polygon(n):
    """Points on a regular n-gon inscribed in unit circle, scaled to unit area."""
    angles = 2 * np.pi * np.arange(n) / n
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return _normalize_unit_area(pts)


def _initial_mixed(n):
    """Mix of boundary and interior points for better minimum triangle area."""
    outer_n = 10
    inner_n = n - outer_n
    angles = 2 * np.pi * np.arange(outer_n) / outer_n
    outer = np.column_stack([np.cos(angles), np.sin(angles)])
    
    inner_angles = 2 * np.pi * np.arange(inner_n) / inner_n + np.pi / inner_n
    inner = 0.4 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    pts = np.vstack([outer, inner])
    return _normalize_unit_area(pts)


def _push_pull_step(pts, alpha=0.1, max_iter=50):
    """
    Push-pull local search: identify the worst (smallest area) triangle
    and push its vertices apart to increase its area.
    """
    pts = pts.copy()
    n = len(pts)
    
    for _ in range(max_iter):
        if time.time() - _push_pull_step._start > DEADLINE - 5:
            break
        
        # Find the worst triangle
        areas = _all_triangle_areas(pts)
        min_idx = np.argmin(areas)
        
        # Convert linear index to (i, j, k)
        # C(n,3) triangles: index maps to combination
        count = 0
        tri = None
        for i in range(n-2):
            for j in range(i+1, n-1):
                for k in range(j+1, n):
                    if count == min_idx:
                        tri = (i, j, k)
                        break
                    count += 1
            if tri:
                break
        if tri:
            break
        
        i, j, k = tri
        pi, pj, pk = pts[i], pts[j], pts[k]
        
        # Compute the centroid of the worst triangle
        centroid = (pi + pj + pk) / 3.0
        
        # Push each vertex away from the centroid
        for idx, p in [(i, pi), (j, pj), (k, pk)]:
            diff = p - centroid
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                direction = diff / norm
                pts[idx] = p + alpha * direction
    
    return _normalize_unit_area(pts)


def _construct_push_pull(init_pts, n_steps=100, alpha_start=0.05, alpha_end=0.001):
    """
    Run push-pull search from an initial configuration.
    """
    _push_pull_step._start = time.time()
    pts = init_pts.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    
    n_steps_actual = min(n_steps, 200)
    
    for step in range(n_steps_actual):
        if time.time() - _push_pull_step._start > DEADLINE - 10:
            break
        
        # Anneal alpha
        alpha = alpha_start * (alpha_end / alpha_start) ** (step / max(1, n_steps_actual - 1))
        
        # Do a push-pull step
        new_pts = _push_pull_step(pts, alpha=alpha, max_iter=10)
        
        # Evaluate
        new_score = _score(new_pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
        
        # Accept with some probability (simulated annealing style)
        if new_score >= best_score or np.random.random() < 0.1:
            pts = new_pts
    
    return best_pts, best_score


def _neg_score(pts_flat):
    """Negative score for minimization."""
    pts = pts_flat.reshape(-1, 2)
    s = _score(pts)
    return -s


def construct():
    start_time = time.time()
    
    # Try multiple initial configurations
    inits = [
        _initial_regular_polygon(N),
        _initial_mixed(N),
    ]
    
    # Also try a few random perturbations
    rng = np.random.RandomState(42)
    for _ in range(3):
        base = _initial_regular_polygon(N)
        noise = rng.normal(0, 0.05, size=base.shape)
        perturbed = base + noise
        inits.append(_normalize_unit_area(perturbed))
    
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Push-pull search from each initial
    for init in inits:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        try:
            cand, s = _construct_push_pull(init, n_steps=50)
            if s > best_score:
                best_score = s
                best_pts = cand
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead polish if time permits
    if best_pts is not None and time.time() - start_time < DEADLINE - 20:
        try:
            x0 = best_pts.flatten()
            res = minimize(_neg_score, x0, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9,
                                  'adaptive': True})
            if res.x is not None:
                cand = res.x.reshape(-1, 2)
                s = _score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand
        except Exception:
            pass
    
    # Phase 3: More push-pull refinement if time permits
    if best_pts is not None and time.time() - start_time < DEADLINE - 10:
        try:
            _push_pull_step._start = time.time()
            for _ in range(30):
                if time.time() - _push_pull_step._start > DEADLINE - 5:
                    break
                new_pts = _push_pull_step(best_pts, alpha=0.01, max_iter=5)
                s = _score(new_pts)
                if s > best_score:
                    best_score = s
                    best_pts = new_pts
        except Exception:
            pass
    
    if best_pts is None:
        best_pts = _initial_regular_polygon(N)
    
    return best_pts
# EVOLVE-BLOCK-END
