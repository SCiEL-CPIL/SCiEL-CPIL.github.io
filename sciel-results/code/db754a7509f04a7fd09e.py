"""Heilbronn convex regions: maximize min triangle area over convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

_DEADLINE = 150.0
_START = time.time()
_SAFETY = 15.0  # seconds before deadline to stop

# Pre-compute triangle indices
TRI_INDICES = np.array(list(combinations(range(N), 3)))
NUM_TRI = len(TRI_INDICES)

def _min_triangle_area_exact(pts):
    """Minimum area of any triangle formed by 3 of the points. Vectorized."""
    p_i = pts[TRI_INDICES[:, 0]]
    p_j = pts[TRI_INDICES[:, 1]]
    p_k = pts[TRI_INDICES[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return np.min(areas)

def _convex_hull_area(pts):
    """Area of the convex hull of the points."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    return _min_triangle_area_exact(pts) / hull_area

def _renormalize_unit_hull(pts):
    """
    Rescales points so that their convex hull has area exactly 1.0.
    Also translates the centroid to (0,0) to reduce parameter space drift.
    Returns None if hull area is invalid.
    """
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return None
    
    # Scale factor: Area scales with scale^2.
    # We want New_Area = 1.0
    # New_Area = scale^2 * Old_Area
    # scale = 1 / sqrt(Old_Area)
    scale = 1.0 / np.sqrt(hull_area)
    
    # Translate to center
    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid
    pts_scaled = pts_centered * scale
    
    return pts_scaled

def _symmetric_ring_init():
    """
    Initialize with a 1-center + 12-outer ring structure.
    This matches the likely optimal geometry for N=13.
    """
    # 1 center point
    center = np.array([0.0, 0.0])
    
    # 12 outer points on a circle
    angles = 2 * np.pi * np.arange(12) / 12
    # Start with a radius that gives unit hull area?
    # Hull area of regular 12-gon with radius R is (1/2) * 12 * R^2 * sin(2pi/12) = 6 * R^2 * sin(pi/6) = 3 * R^2.
    # We want area 1, so R^2 = 1/3 => R = 1/sqrt(3).
    # But the center point doesn't affect the hull area if it's inside.
    # So we can just normalize afterwards.
    radius = 1.0
    outer = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    
    pts = np.vstack([center, outer])
    
    # Add small perturbation to break symmetry for optimizer
    rng = np.random.RandomState(42)
    pts += rng.randn(N, 2) * 0.01
    
    return _renormalize_unit_hull(pts)

def _objective(x):
    """
    Objective function for minimization (we minimize negative score).
    x: flattened array of 2N coordinates.
    """
    # Check time limit
    if time.time() - _START > _DEADLINE - _SAFETY:
        return 0.0
        
    pts = x.reshape(N, 2)
    
    # Renormalize to fix scale
    pts_norm = _renormalize_unit_hull(pts)
    if pts_norm is None:
        return 0.0
        
    # Calculate score on normalized points
    # Since hull area is 1, score is just min_triangle_area
    min_area = _min_triangle_area_exact(pts_norm)
    
    # Return negative score for minimization
    return -min_area

def construct():
    global _START, _DEADLINE
    elapsed = time.time() - _START
    remaining = _DEADLINE - _SAFETY - elapsed
    
    if remaining < 10.0:
        # Not enough time for optimization, return best static guess
        return _symmetric_ring_init()

    # Start with symmetric ring configuration
    init_pts = _symmetric_ring_init()
    if init_pts is None:
        # Fallback to random
        init_pts = np.random.rand(N, 2) * 4 - 2
        init_pts = _renormalize_unit_hull(init_pts)
    
    best_pts = init_pts
    best_score = _score(best_pts)
    
    # Strategy: Multiple restarts with Nelder-Mead
    # Nelder-Mead is effective for this low-dimensional non-smooth problem if started from a good symmetric init.
    # We will try a few different perturbations and keep the best.
    
    rng = np.random.RandomState(42)
    
    # Time budget for each restart
    # We have ~135s. Let's do 5 restarts of ~25s each?
    # Or just run one long one? Nelder-Mead can get stuck in local minima.
    # Symmetric init is likely close to global optimum.
    
    # Let's do a sequence of Nelder-Mead runs with increasing perturbation.
    # 1. Fine tune the symmetric init.
    # 2. If time remains, try a slightly perturbed version.
    
    def run_nm(init_x, max_iter, time_limit):
        """Run Nelder-Mead with time limit."""
        start_time = time.time()
        
        def obj(x):
            if time.time() - _START > _DEADLINE - _SAFETY:
                return 0.0
            pts = x.reshape(N, 2)
            pts_norm = _renormalize_unit_hull(pts)
            if pts_norm is None:
                return 0.0
            return -_min_triangle_area_exact(pts_norm)
        
        try:
            res = minimize(obj, init_x, method='Nelder-Mead',
                           options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            return res.x
        except Exception:
            return init_x

    # Run 1: Fine tune symmetric init
    x0 = best_pts.flatten()
    x_res = run_nm(x0, 2000, 30.0)
    if x_res is not None:
        pts_res = x_res.reshape(N, 2)
        pts_res = _renormalize_unit_hull(pts_res)
        if pts_res is not None:
            s_res = _score(pts_res)
            if s_res > best_score:
                best_pts = pts_res
                best_score = s_res
    
    # Run 2: Perturbed symmetric init (slightly larger perturbation)
    if time.time() - _START < _DEADLINE - _SAFETY - 30.0:
        # Create a perturbed version
        init_pts2 = _symmetric_ring_init()
        # Apply a slightly larger random perturbation
        rng2 = np.random.RandomState(123)
        init_pts2 += rng2.randn(N, 2) * 0.05
        init_pts2 = _renormalize_unit_hull(init_pts2)
        
        if init_pts2 is not None:
            x0_2 = init_pts2.flatten()
            x_res2 = run_nm(x0_2, 2000, 30.0)
            if x_res2 is not None:
                pts_res2 = x_res2.reshape(N, 2)
                pts_res2 = _renormalize_unit_hull(pts_res2)
                if pts_res2 is not None:
                    s_res2 = _score(pts_res2)
                    if s_res2 > best_score:
                        best_pts = pts_res2
                        best_score = s_res2
    
    # Run 3: Another perturbation if time permits
    if time.time() - _START < _DEADLINE - _SAFETY - 30.0:
        init_pts3 = _symmetric_ring_init()
        rng3 = np.random.RandomState(456)
        init_pts3 += rng3.randn(N, 2) * 0.1
        init_pts3 = _renormalize_unit_hull(init_pts3)
        
        if init_pts3 is not None:
            x0_3 = init_pts3.flatten()
            x_res3 = run_nm(x0_3, 2000, 30.0)
            if x_res3 is not None:
                pts_res3 = x_res3.reshape(N, 2)
                pts_res3 = _renormalize_unit_hull(pts_res3)
                if pts_res3 is not None:
                    s_res3 = _score(pts_res3)
                    if s_res3 > best_score:
                        best_pts = pts_res3
                        best_score = s_res3

    return best_pts
# EVOLVE-BLOCK-END
