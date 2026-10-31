"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# Precompute all triples
TRIPLES = list(combinations(range(N), 3))
TRIPLES_ARR = np.array(TRIPLES, dtype=np.int64)  # shape (286, 3)


def _min_triangle_area(pts):
    """Minimum triangle area over all triples of points."""
    # pts: (N, 2)
    # For each triple (i,j,k), area = 0.5 * |cross(pts[j]-pts[i], pts[k]-pts[i])|
    pi = pts[TRIPLES_ARR[:, 0]]  # (286, 2)
    pj = pts[TRIPLES_ARR[:, 1]]
    pk = pts[TRIPLES_ARR[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _convex_hull_area(pts):
    """Area of convex hull of points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 1.0


def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_area


def _negative_score(pts):
    """Negative score for minimization."""
    return -_score(pts)


def _regular_polygon_angles(n, offset=0.0):
    """Regular polygon angles with offset."""
    return offset + 2.0 * np.pi * np.arange(n) / n


def _optimize_from_initial(pts_init, deadline, verbose=False):
    """Optimize from an initial point configuration using L-BFGS-B."""
    best_pts = pts_init.copy()
    best_score = _score(best_pts)
    
    # Flatten to 2D parameter vector
    x0 = best_pts.flatten()
    
    def objective(x):
        if time.time() > deadline:
            return 1e10  # Penalize if past deadline
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    # Use L-BFGS-B with bounds (keep points in a reasonable box)
    bounds = [(-10, 10)] * (2 * N)
    
    try:
        res = minimize(
            objective, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if time.time() < deadline:
            pts_opt = res.x.reshape(-1, 2)
            s = _score(pts_opt)
            if s > best_score:
                best_pts = pts_opt
                best_score = s
    except Exception:
        pass
    
    return best_pts, best_score


def _optimize_radial_angular(deadline):
    """Optimize using radial/angular parameterization."""
    # Start with regular polygon on unit circle
    angles = _regular_polygon_angles(N)
    radii = np.ones(N)
    
    # Parameter vector: [r0, r1, ..., r12, theta0, theta1, ..., theta12]
    x0 = np.concatenate([radii, angles])
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        radii = x[:N]
        angles = x[N:]
        # Ensure positive radii
        radii = np.maximum(radii, 0.01)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return -_score(pts)
    
    bounds = [(0.01, 10.0)] * N + [(-10.0, 10.0)] * N
    
    best_pts = None
    best_score = 0.0
    
    try:
        res = minimize(
            objective, x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'maxfun': 1500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if time.time() < deadline:
            radii = np.maximum(res.x[:N], 0.01)
            angles = res.x[N:]
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts
    except Exception:
        pass
    
    return best_pts, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_pts = None
    best_score = 0.0
    
    # Strategy 1: Regular polygon with slight perturbations
    for offset in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
        if time.time() > deadline - 5.0:
            break
        angles = _regular_polygon_angles(N, offset)
        radii = np.ones(N)
        pts_init = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        # Add small random perturbation
        rng = np.random.RandomState(int(offset * 1000) + 42)
        pts_init += 0.01 * rng.randn(N, 2)
        
        pts_opt, score = _optimize_from_initial(pts_init, deadline)
        if score > best_score:
            best_score = score
            best_pts = pts_opt
    
    # Strategy 2: Radial/angular optimization
    if time.time() < deadline - 10.0:
        pts_rad, score_rad = _optimize_radial_angular(deadline)
        if pts_rad is not None and score_rad > best_score:
            best_score = score_rad
            best_pts = pts_rad
    
    # Strategy 3: More diverse initial configurations
    if time.time() < deadline - 15.0:
        # Golden ratio spacing
        golden = (1 + np.sqrt(5)) / 2
        for trial in range(5):
            if time.time() > deadline - 5.0:
                break
            rng = np.random.RandomState(trial * 7 + 13)
            # Perturbed regular polygon
            angles = _regular_polygon_angles(N, rng.uniform(0, 2*np.pi))
            radii = 1.0 + 0.1 * rng.randn(N)
            radii = np.maximum(radii, 0.1)
            pts_init = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            
            pts_opt, score = _optimize_from_initial(pts_init, deadline)
            if score > best_score:
                best_score = score
                best_pts = pts_opt
    
    # Final: if we still have time, do one more round of refinement
    if best_pts is not None and time.time() < deadline - 3.0:
        pts_final, score_final = _optimize_from_initial(best_pts, deadline)
        if score_final > best_score:
            best_score = score_final
            best_pts = pts_final
    
    if best_pts is None:
        # Fallback: regular polygon
        angles = _regular_polygon_angles(N)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts
# EVOLVE-BLOCK-END
