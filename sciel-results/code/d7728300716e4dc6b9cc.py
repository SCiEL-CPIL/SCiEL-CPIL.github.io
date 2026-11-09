import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _triangle_areas(points):
    """Compute areas of all triangles formed by 3 points. Returns array of shape (286,)."""
    n = len(points)
    i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
    i, j, k = i.T.ravel(), j.T.ravel(), k.T.ravel()
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    areas = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1]) +
        p2[:, 0] * (p3[:, 1] - p1[:, 1]) +
        p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    return areas

def _score(points):
    """Compute A_min / A_hull. Returns 0 if invalid."""
    try:
        if np.any(~np.isfinite(points)):
            return 0.0
        if len(points) != 13 or points.shape != (13, 2):
            return 0.0
        hull = ConvexHull(points)
        a_hull = hull.volume
        if a_hull <= 0:
            return 0.0
        areas = _triangle_areas(points)
        a_min = np.min(areas)
        if a_min <= 0:
            return 0.0
        return a_min / a_hull
    except Exception:
        return 0.0

def _soft_min_penalty(pts, temperature=0.05):
    """
    Smooth approximation of the minimum triangle area using soft-min.
    Returns -soft_min(areas) so we can minimize it.
    """
    areas = _triangle_areas(pts)
    min_val = np.min(areas)
    if min_val < 1e-10:
        return min_val
    shifted = areas - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    soft_min = min_val - temperature * log_sum
    return -soft_min

def _objective_lbfgs(pts_flat, temperature=0.05):
    """Objective for L-BFGS-B: minimize -soft_min + penalty for hull area."""
    n = 13
    pts = pts_flat.reshape(n, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    
    penalty = _soft_min_penalty(pts, temperature)
    
    try:
        hull = ConvexHull(pts)
        a_hull = hull.volume
        if not np.isfinite(a_hull) or a_hull <= 0:
            return 1e10
    except Exception:
        return 1e10
    
    return penalty + 1.0 * np.log(a_hull)

def _objective_nm(pts_flat):
    """Objective for Nelder-Mead: -min_triangle_area."""
    points = pts_flat.reshape(13, 2)
    if np.any(~np.isfinite(points)):
        return 1e6
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    if min_area == 0.0:
        return 1e6
    return -min_area

def _hexagonal_seed():
    """Create a hexagonal lattice seed for 13 points."""
    pts = []
    pts.append([0.0, 0.0])
    r1 = 0.5
    for i in range(6):
        angle = i * np.pi / 3
        pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    r2 = 1.0
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    return np.array(pts)

def _ring_seed():
    """Create concentric ring seed."""
    pts = []
    pts.append([0.0, 0.0])
    r1 = 0.4
    for i in range(5):
        angle = i * 2 * np.pi / 5
        pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    r2 = 0.9
    for i in range(7):
        angle = i * 2 * np.pi / 7
        pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    return np.array(pts)

def _spiral_seed():
    """Create a spiral seed."""
    pts = []
    n = 13
    for i in range(n):
        t = i / (n - 1)
        r = 0.1 + 0.9 * t
        angle = t * 4 * np.pi
        pts.append([r * np.cos(angle), r * np.sin(angle)])
    return np.array(pts)

def _regular_polygon_seed():
    """Regular 13-gon."""
    angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _jittered_grid_seed(rng):
    """Jittered grid seed."""
    x = np.linspace(-1, 1, 4)
    y = np.linspace(-1, 1, 4)
    xx, yy = np.meshgrid(x, y)
    pts = np.column_stack((xx.ravel(), yy.ravel()))[:13]
    pts += rng.normal(0, 0.1, size=pts.shape)
    return pts

def _two_ring_seed(rng):
    """Two concentric rings with jitter."""
    n1 = 7
    n2 = 6
    angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False) + rng.uniform(0, 2*np.pi)
    angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + rng.uniform(0, 2*np.pi)
    r1 = 1.0
    r2 = 0.5
    pts1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
    pts2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
    return np.vstack([pts1, pts2])

def _random_disk_seed(rng):
    """Random points in unit disk."""
    n = 13
    r = np.sqrt(rng.random(n))
    theta = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack((r * np.cos(theta), r * np.sin(theta)))

def _coordinate_descent(pts, max_iters=200, step=0.02, deadline=None):
    """
    Coordinate descent to refine points using actual score.
    """
    n = 13
    pts = pts.copy()
    best_score = _score(pts)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step),
                           (step*0.5, step*0.5), (-step*0.5, step*0.5),
                           (step*0.5, -step*0.5), (-step*0.5, -step*0.5)]:
                new_pts = pts.copy()
                new_pts[i, 0] += dx
                new_pts[i, 1] += dy
                s = _score(new_pts)
                if s > best_score:
                    pts = new_pts
                    best_score = s
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-7:
                break
    
    return pts, best_score

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the minimum triangle area
    relative to the convex hull area (Heilbronn problem).
    
    Uses multiple diverse seeds with L-BFGS-B (smooth soft-min) + Nelder-Mead
    (non-smooth refinement) + coordinate descent (final polish).
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    rng = np.random.default_rng(42)
    
    # Generate diverse seeds
    seeds = [
        _hexagonal_seed(),
        _ring_seed(),
        _spiral_seed(),
        _regular_polygon_seed(),
        _jittered_grid_seed(rng),
        _two_ring_seed(rng),
        _random_disk_seed(rng),
    ]
    # Add more random seeds
    for _ in range(4):
        seeds.append(_random_disk_seed(rng))
    
    best_pts = None
    best_score = 0.0
    
    for seed_idx, seed in enumerate(seeds):
        if time.time() > deadline - 30:
            break
        
        # Normalize seed
        seed = seed - np.mean(seed, axis=0)
        scale = np.max(np.abs(seed))
        if scale > 0:
            seed = seed / scale * 0.5
        
        x0 = seed.flatten()
        
        # Phase 1: L-BFGS-B with smooth soft-min
        try:
            res_lbfgs = minimize(
                _objective_lbfgs,
                x0,
                method='L-BFGS-B',
                args=(0.05,),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res_lbfgs.fun < 1e8:
                pts = res_lbfgs.x.reshape(13, 2)
                if np.all(np.isfinite(pts)):
                    s = _score(pts)
                    if s > best_score:
                        best_score = s
                        best_pts = pts.copy()
        except Exception:
            pass
        
        # Phase 2: Nelder-Mead refinement from L-BFGS-B result (or seed if L-BFGS failed)
        if time.time() > deadline - 20:
            break
        
        # Use L-BFGS result if available, else seed
        nm_start = x0
        if 'res_lbfgs' in locals() and res_lbfgs.fun < 1e8:
            nm_start = res_lbfgs.x
        
        try:
            res_nm = minimize(
                _objective_nm,
                nm_start,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-9, 'fatol': 1e-11, 'maxfev': 3000}
            )
            if res_nm.fun < 1e5:
                pts = res_nm.x.reshape(13, 2)
                if np.all(np.isfinite(pts)):
                    s = _score(pts)
                    if s > best_score:
                        best_score = s
                        best_pts = pts.copy()
        except Exception:
            pass
    
    # Phase 3: Coordinate descent polish on best candidate
    if best_pts is not None and time.time() < deadline - 10:
        refined_pts, refined_score = _coordinate_descent(
            best_pts, max_iters=150, step=0.03, deadline=deadline
        )
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts.copy()
    
    # Final polish with smaller steps
    if best_pts is not None and time.time() < deadline - 5:
        refined_pts, refined_score = _coordinate_descent(
            best_pts, max_iters=100, step=0.01, deadline=deadline
        )
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts.copy()
    
    # Fallback if no valid points found
    if best_pts is None or _score(best_pts) <= 0:
        best_pts = _hexagonal_seed()
        best_pts = best_pts - np.mean(best_pts, axis=0)
        scale = np.max(np.abs(best_pts))
        if scale > 0:
            best_pts = best_pts / scale * 0.5
    
    return best_pts
