"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import itertools
import math

N = 13

# EVOLVE-BLOCK-START

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas for n points."""
    n = len(pts)
    areas = np.empty((n, n, n), dtype=np.float64)
    areas.fill(0.0)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |cross product|
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                areas[i, j, k] = area
    return areas

def _min_triangle_area(pts):
    """Compute the minimum triangle area among all triples."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    """Compute the area of the convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(pts):
    """Compute the score: min triangle area / hull area."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_a

def _negative_score(pts):
    """Negative of score for minimization."""
    return -_score(pts)

def _halton_sequence(n, base):
    """Generate n points in the Halton sequence for a given base."""
    seq = []
    for i in range(1, n + 1):
        f = 1.0
        r = 0.0
        j = i
        while j > 0:
            j, rem = divmod(j, base)
            f /= base
            r += f * rem
        seq.append(r)
    return seq

def _halton_points(n, base1=2, base2=3):
    """Generate n 2D Halton points."""
    x = _halton_sequence(n, base1)
    y = _halton_sequence(n, base2)
    return np.column_stack([x, y])

def _local_refine(pts, max_iter=200, step_size=0.01, deadline=None):
    """Local refinement using coordinate descent."""
    pts = pts.copy().astype(np.float64)
    best_score = _score(pts)
    current_score = best_score
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for idx in range(N):
            for dim in range(2):
                best_val = pts[idx, dim]
                best_s = current_score
                
                # Try small perturbations
                for delta in [-step_size, -step_size/2, step_size/2, step_size]:
                    trial = pts.copy()
                    trial[idx, dim] += delta
                    s = _score(trial)
                    if s > best_s:
                        best_s = s
                        best_val = trial[idx, dim]
                
                if best_val != pts[idx, dim]:
                    pts[idx, dim] = best_val
                    current_score = best_s
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
        
        if not improved:
            break
        # Reduce step size gradually
        step_size *= 0.7
    
    return pts, best_score

def _differential_evolution_search(deadline):
    """Use differential evolution for global search."""
    # Bounds: points in [0, 1] x [0, 1]
    bounds = [(0, 1)] * (2 * N)
    
    # Use a smaller population and fewer iterations to fit within time
    result = differential_evolution(
        _negative_score,
        bounds=bounds,
        seed=42,
        maxiter=50,
        popsize=15,
        tol=1e-6,
        mutation=(0.3, 1.0),
        recombination=0.8,
        polish=True,
        init='latinhypercube'
    )
    
    if result.success or result.fun is not None:
        best_pts = result.x.reshape(N, 2)
        best_score = _score(best_pts)
        return best_pts, best_score
    
    return None, 0.0

def _multi_start_local(deadline):
    """Multiple starts with local refinement."""
    best_pts = None
    best_score = 0.0
    
    # Start with Halton sequence
    pts_halton = _halton_points(N)
    refined, score = _local_refine(pts_halton, max_iter=100, step_size=0.02, deadline=deadline)
    if score > best_score:
        best_score = score
        best_pts = refined.copy()
    
    # Try a few random starts
    rng = np.random.RandomState(42)
    for trial in range(5):
        if time.time() > deadline:
            break
        pts_random = rng.rand(N, 2)
        refined, score = _local_refine(pts_random, max_iter=50, step_size=0.03, deadline=deadline)
        if score > best_score:
            best_score = score
            best_pts = refined.copy()
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    # Leave 15 seconds safety margin
    deadline = start_time + 105
    
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Differential evolution for global search
    try:
        de_pts, de_score = _differential_evolution_search(deadline)
        if de_pts is not None and de_score > best_score:
            best_score = de_score
            best_pts = de_pts
    except Exception:
        pass
    
    if time.time() > deadline:
        # Return best found
        if best_pts is not None:
            return best_pts
        return _halton_points(N)
    
    # Phase 2: Multi-start local refinement
    try:
        ms_pts, ms_score = _multi_start_local(deadline)
        if ms_pts is not None and ms_score > best_score:
            best_score = ms_score
            best_pts = ms_pts
    except Exception:
        pass
    
    if time.time() > deadline:
        if best_pts is not None:
            return best_pts
        return _halton_points(N)
    
    # Phase 3: Final local refinement on the best candidate
    if best_pts is not None:
        try:
            refined, final_score = _local_refine(best_pts, max_iter=150, step_size=0.005, deadline=deadline)
            if final_score > best_score:
                best_score = final_score
                best_pts = refined
        except Exception:
            pass
    
    if best_pts is not None:
        return best_pts
    
    # Fallback: Halton sequence
    return _halton_points(N)

# EVOLVE-BLOCK-END
