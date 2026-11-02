"""Heilbronn convex regions - maximize min triangle area / hull area for 13 points."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 13


def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples of points."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_area = float('inf')
    # Vectorized approach: compute all pairwise differences and cross products
    for i in range(n):
        for j in range(i + 1, n):
            v1 = pts[j] - pts[i]
            for k in range(j + 1, n):
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area


def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 0:
            return 0.0
    except Exception:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area


def _neg_score(pts):
    """Negative score for minimization."""
    return -_score_points(pts)


def _regular_polygon_with_center(n, radius=1.0):
    """Create a regular (n-1)-gon with one point at center."""
    angles = np.linspace(0, 2 * np.pi, n - 1, endpoint=False)
    pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    center = np.array([[0.0, 0.0]])
    return np.vstack([pts, center])


def _two_rings(n):
    """Two concentric rings of points."""
    # Split points between two rings
    outer = n // 2  # 6 outer
    inner = n - outer  # 7 inner
    angles_out = np.linspace(0, 2 * np.pi, outer, endpoint=False)
    angles_in = np.linspace(0, 2 * np.pi, inner, endpoint=False) + np.pi / inner
    r_out = 1.0
    r_in = 0.5
    pts_out = np.column_stack([r_out * np.cos(angles_out), r_out * np.sin(angles_out)])
    pts_in = np.column_stack([r_in * np.cos(angles_in), r_in * np.sin(angles_in)])
    return np.vstack([pts_out, pts_in])


def _jittered_regular(n, jitter=0.05, seed=42):
    """Regular polygon with jitter."""
    rng = np.random.RandomState(seed)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radii = 1.0 + rng.uniform(-jitter, jitter, n)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts


def _jittered_two_rings(n, jitter=0.05, seed=42):
    """Two rings with jitter."""
    rng = np.random.RandomState(seed)
    outer = n // 2
    inner = n - outer
    angles_out = np.linspace(0, 2 * np.pi, outer, endpoint=False) + rng.uniform(-jitter, jitter, outer)
    angles_in = np.linspace(0, 2 * np.pi, inner, endpoint=False) + np.pi / inner + rng.uniform(-jitter, jitter, inner)
    r_out = 1.0 + rng.uniform(-jitter, jitter, outer)
    r_in = 0.5 + rng.uniform(-jitter, jitter, inner)
    pts_out = np.column_stack([r_out * np.cos(angles_out), r_out * np.sin(angles_out)])
    pts_in = np.column_stack([r_in * np.cos(angles_in), r_in * np.sin(angles_in)])
    return np.vstack([pts_out, pts_in])


def _soft_min_triangle_area(pts, gamma=10.0):
    """Soft minimum of triangle areas for differentiability."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            v1 = pts[j] - pts[i]
            for k in range(j + 1, n):
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                areas.append(area)
    areas = np.array(areas)
    if len(areas) == 0:
        return 0.0
    # Soft min: -log-sum-exp(-gamma * areas) / gamma
    scaled = -gamma * areas
    max_val = np.max(scaled)
    soft_min = -max_val / gamma - np.log(np.sum(np.exp(scaled - max_val))) / gamma
    return soft_min


def _soft_min_score(pts, gamma=10.0):
    """Soft min triangle area / hull area."""
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 0:
            return 0.0
    except Exception:
        return 0.0
    min_area = _soft_min_triangle_area(pts, gamma)
    return min_area / hull_area


def _neg_soft_score(pts, gamma=10.0):
    """Negative soft score for minimization."""
    return -_soft_min_score(pts, gamma)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Generate diverse initial configurations
    inits = []
    inits.append(('reg_center', _regular_polygon_with_center(N, 1.0)))
    inits.append(('two_rings', _two_rings(N)))
    for seed in range(5):
        inits.append((f'jitter_reg_{seed}', _jittered_regular(N, 0.05, seed)))
        inits.append((f'jitter_2r_{seed}', _jittered_two_rings(N, 0.05, seed)))
    
    # Score all initial configurations
    scored_inits = []
    for name, pts in inits:
        score = _score_points(pts)
        scored_inits.append((score, name, pts))
    
    # Sort by score descending
    scored_inits.sort(key=lambda x: -x[0])
    
    best_score = scored_inits[0][0]
    best_pts = scored_inits[0][2]
    
    # Try to improve each initial configuration using local optimization
    for score, name, pts in scored_inits:
        if time.time() > deadline:
            break
        
        if score <= best_score:
            continue
        
        # Use L-BFGS-B with soft min objective
        x0 = pts.flatten()
        
        try:
            result = minimize(
                _neg_soft_score,
                x0,
                args=(15.0,),  # gamma
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < -score:
                improved_pts = result.x.reshape(-1, 2)
                improved_score = _score_points(improved_pts)
                if improved_score > best_score:
                    best_score = improved_score
                    best_pts = improved_pts
        except Exception:
            pass
        
        # Try Nelder-Mead refinement
        if time.time() > deadline:
            break
        
        try:
            result2 = minimize(
                _neg_score,
                pts,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if result2.fun < -score:
                improved_pts = result2.x
                improved_score = _score_points(improved_pts)
                if improved_score > best_score:
                    best_score = improved_score
                    best_pts = improved_pts
        except Exception:
            pass
    
    # Final polish with L-BFGS-B on the best point
    if time.time() < deadline - 5.0:
        try:
            x0 = best_pts.flatten()
            result = minimize(
                _neg_soft_score,
                x0,
                args=(20.0,),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.fun < -best_score:
                improved_pts = result.x.reshape(-1, 2)
                improved_score = _score_points(improved_pts)
                if improved_score > best_score:
                    best_score = improved_score
                    best_pts = improved_pts
        except Exception:
            pass
    
    return best_pts
