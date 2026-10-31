"""Heilbronn convex regions - 13 points maximizing min triangle area / hull area.

Structural change: use a grid-based representation with affine optimization,
plus a dedicated local search that directly targets the smallest triangles.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

N = 13

def _min_triangle_area(pts):
    """Compute minimum area over all triples of points (vectorized-ish)."""
    n = pts.shape[0]
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10

def score_function(pts):
    ha = _hull_area(pts)
    if ha < 1e-10:
        return 0.0
    return _min_triangle_area(pts) / ha

def _grid_start():
    """3x5 grid with center column shifted to break symmetry."""
    xs = [0.0, 1.0, 2.0]
    ys = [0.0, 1.0, 2.0, 3.0, 4.0]
    pts = []
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            if yi == 2 and xi == 1:
                continue  # skip center to make 12, then add 1
            pts.append([x, y])
    pts.append([1.0, 2.0])  # center point
    pts = np.array(pts, dtype=float)
    # Center and normalize
    c = pts.mean(axis=0)
    pts -= c
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts /= scale
    return pts

def _hex_start():
    """Hexagonal arrangement: center + 6 ring + 6 outer."""
    pts = [[0.0, 0.0]]
    for i in range(6):
        a = i * np.pi / 3
        pts.append([0.6 * np.cos(a), 0.6 * np.sin(a)])
    for i in range(6):
        a = i * np.pi / 3 + np.pi / 6
        pts.append([1.0 * np.cos(a), 1.0 * np.sin(a)])
    pts = np.array(pts, dtype=float)
    c = pts.mean(axis=0)
    pts -= c
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts /= scale
    return pts

def _circular_start():
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    return pts

def _random_start(seed):
    rng = np.random.RandomState(seed)
    pts = rng.rand(N, 2) * 2 - 1
    c = pts.mean(axis=0)
    pts -= c
    scale = np.max(np.abs(pts))
    if scale > 0:
        pts /= scale
    return pts

def _local_search(pts, deadline, max_rounds=200):
    """
    Local search that identifies the smallest triangles and nudges
    the involved points to increase their area.
    """
    current = pts.copy()
    best_score = score_function(current)
    best_pts = current.copy()
    
    for rnd in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Find all triangles and their areas
        n = current.shape[0]
        tri_areas = []
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    v1 = current[j] - current[i]
                    v2 = current[k] - current[i]
                    area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                    tri_areas.append((area, i, j, k))
        
        tri_areas.sort()
        min_area = tri_areas[0][0]
        if min_area < 1e-12:
            break
        
        # Take the k smallest triangles and try to increase them
        k_smallest = min(5, len(tri_areas))
        step = 0.01 * (min_area / best_score if best_score > 0 else 0.01)
        
        improved = False
        for area, i, j, k in tri_areas[:k_smallest]:
            if time.time() > deadline:
                break
            
            # For each point in the triangle, try nudging it away from the other two
            for idx in [i, j, k]:
                others = [x for x in [i, j, k] if x != idx]
                # Compute direction away from the midpoint of the other two
                mid = (current[others[0]] + current[others[1]]) / 2
                direction = current[idx] - mid
                norm = np.linalg.norm(direction)
                if norm < 1e-12:
                    direction = np.array([1.0, 0.0])
                    norm = 1.0
                direction /= norm
                
                # Try a small nudge
                for s in [step, step * 0.5, step * 2]:
                    trial = current.copy()
                    trial[idx] += direction * s
                    trial_score = score_function(trial)
                    if trial_score > best_score:
                        best_score = trial_score
                        best_pts = trial.copy()
                        current = trial.copy()
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try random perturbations
            rng = np.random.RandomState(rnd + 1000)
            for _ in range(10):
                if time.time() > deadline:
                    break
                trial = current.copy()
                trial += rng.randn(N, 2) * 0.005
                trial_score = score_function(trial)
                if trial_score > best_score:
                    best_score = trial_score
                    best_pts = trial.copy()
                    current = trial.copy()
                    improved = True
                    break
            if not improved:
                break
    
    return best_pts, best_score

def _affine_optimize(pts, deadline):
    """
    Optimize an affine transformation applied to a base configuration.
    Parameters: [a, b, c, d, tx, ty] where transformed = [[a,b],[c,d]] @ pts + [tx, ty]
    """
    base = pts.copy()
    
    def neg_score_affine(params):
        a, b, c, d, tx, ty = params
        M = np.array([[a, b], [c, d]])
        t = np.array([tx, ty])
        transformed = base @ M.T + t
        return -score_function(transformed)
    
    # Start from identity
    x0 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    
    try:
        result = minimize(neg_score_affine, x0, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        if time.time() > deadline:
            return pts, score_function(pts)
        
        a, b, c, d, tx, ty = result.x
        M = np.array([[a, b], [c, d]])
        t = np.array([tx, ty])
        transformed = base @ M.T + t
        sc = score_function(transformed)
        if sc > score_function(pts):
            return transformed, sc
    except Exception:
        pass
    
    return pts, score_function(pts)

def construct():
    deadline = time.time() + 115  # Safety margin before 150s limit
    
    starts = [
        _grid_start(),
        _hex_start(),
        _circular_start(),
        _random_start(42),
        _random_start(123),
        _random_start(999),
        _random_start(777),
        _random_start(555),
    ]
    
    best_pts = starts[0]
    best_score = score_function(best_pts)
    
    for i, start in enumerate(starts):
        if time.time() > deadline:
            break
        
        # Affine optimization
        aff_pts, aff_score = _affine_optimize(start, deadline)
        if aff_score > best_score:
            best_score = aff_score
            best_pts = aff_pts
        
        # Local search from the best of start and affine
        base_for_ls = aff_pts if aff_score > score_function(start) else start
        ls_pts, ls_score = _local_search(base_for_ls, deadline, max_rounds=150)
        if ls_score > best_score:
            best_score = ls_score
            best_pts = ls_pts
        
        # Another round of affine after local search
        if time.time() < deadline - 15:
            aff2_pts, aff2_score = _affine_optimize(best_pts, deadline)
            if aff2_score > best_score:
                best_score = aff2_score
                best_pts = aff2_pts
    
    # Final local search
    if time.time() < deadline - 5:
        final_pts, final_score = _local_search(best_pts, deadline, max_rounds=100)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts
    
    return best_pts
