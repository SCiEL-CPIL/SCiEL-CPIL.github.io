"""Heilbronn triangles: place 11 points in a unit-area equilateral triangle to maximize
the smallest triangle area formed by any three points."""

import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize, differential_evolution

N = 11
# Unit-area equilateral triangle: vertices at (0,0), (2,0), (1, sqrt(3))
# Area = 0.5 * 2 * sqrt(3) = sqrt(3)
# We'll work in this coordinate system and divide by sqrt(3) at the end.

# Triangle vertices
T0 = np.array([0.0, 0.0])
T1 = np.array([2.0, 0.0])
T2 = np.array([1.0, np.sqrt(3.0)])

# Pre-compute all C(11,3) = 165 triangle index combinations
TRI_INDICES = np.array(list(combinations(range(N), 3)))

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the N points.
    pts: (N, 2) array of (x, y) coordinates.
    Returns the minimum area (not doubled)."""
    p1 = pts[TRI_INDICES[:, 0]]  # (165, 2)
    p2 = pts[TRI_INDICES[:, 1]]
    p3 = pts[TRI_INDICES[:, 2]]
    
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    dx1 = p2[:, 0] - p1[:, 0]
    dy1 = p2[:, 1] - p1[:, 1]
    dx2 = p3[:, 0] - p1[:, 0]
    dy2 = p3[:, 1] - p1[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _objective(pts_flat):
    """Objective function for minimization (negative of min area)."""
    pts = pts_flat.reshape(N, 2)
    return -_min_triangle_area(pts)

def _point_in_triangle(p, a, b, c):
    """Check if point p is inside or on triangle abc using barycentric coordinates."""
    x, y = p
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    
    # Barycentric coordinates
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(denom) < 1e-12:
        return False
    a_coord = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
    b_coord = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
    c_coord = 1.0 - a_coord - b_coord
    
    return (a_coord >= -1e-9) and (b_coord >= -1e-9) and (c_coord >= -1e-9)

def _project_to_triangle(p):
    """Project point p onto the triangle if it's outside. Simple approach: clamp to triangle."""
    x, y = p
    # Check if inside
    if _point_in_triangle(p, T0, T1, T2):
        return p
    
    # If outside, find closest point on triangle boundary
    # Simple approach: move towards centroid
    centroid = (T0 + T1 + T2) / 3.0
    # Binary search for point on segment from p to centroid that is inside
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2.0
        test = p + mid * (centroid - p)
        if _point_in_triangle(test, T0, T1, T2):
            lo = mid
        else:
            hi = mid
    return p + lo * (centroid - p)

def _enforce_constraints(pts_flat):
    """Project points to be inside the triangle."""
    pts = pts_flat.reshape(N, 2)
    for i in range(N):
        pts[i] = _project_to_triangle(pts[i])
    return pts.flatten()

def _tammes_init(seed=0):
    """Generate initial points using a structured placement.
    Use a grid-like arrangement adapted to the triangle."""
    rng = np.random.RandomState(seed)
    
    # Start with a structured pattern: rows of points
    # For 11 points in an equilateral triangle, use a triangular lattice
    # Row 0: 1 point (top)
    # Row 1: 2 points
    # Row 2: 3 points
    # Row 3: 4 points
    # Row 4: 1 point (or distribute differently)
    # Total: 1+2+3+4 = 10, need 1 more
    
    # Alternative: use 4 rows: 1, 2, 3, 5 or 2, 3, 3, 3
    # Let's use a more uniform approach with some randomness
    
    points = []
    
    # Place points on a triangular grid with jitter
    # Scale: triangle has base 2, height sqrt(3)
    # Use 4 rows
    rows = [1, 2, 3, 5]  # Total = 11
    total_rows = len(rows)
    
    for r_idx, count in enumerate(rows):
        # y position: from top to bottom
        # Top vertex at y=sqrt(3), bottom at y=0
        y = np.sqrt(3.0) * (1.0 - r_idx / total_rows)
        # x positions: centered, spread across the width at this y
        # Width at height y: the triangle narrows linearly
        # At y=0, width=2; at y=sqrt(3), width=0
        width = 2.0 * y / np.sqrt(3.0)
        if count == 1:
            x_pos = np.array([1.0])  # center
        else:
            x_pos = np.linspace(1.0 - width/2.0, 1.0 + width/2.0, count)
        
        for x in x_pos:
            # Add small jitter
            jx = rng.uniform(-0.05, 0.05)
            jy = rng.uniform(-0.05, 0.05)
            points.append([x + jx, y + jy])
    
    pts = np.array(points)
    # Project to triangle
    for i in range(N):
        pts[i] = _project_to_triangle(pts[i])
    return pts.flatten()

def _random_init(seed=0):
    """Generate random initial points inside the triangle."""
    rng = np.random.RandomState(seed)
    points = []
    while len(points) < N:
        u = rng.random()
        v = rng.random()
        if u + v < 1.0:
            x = 2.0 * (u + 0.5 * v)
            y = np.sqrt(3.0) * v
            points.append([x, y])
    return np.array(points).flatten()

def construct():
    """Main entry point. Returns array of shape (N, 2) with points in the unit-area triangle."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin from 150s limit
    
    best_points = None
    best_score = -np.inf
    
    # Strategy 1: Differential evolution for global exploration
    # Then local refinement with L-BFGS-B
    
    # First, try a few structured/random initializations with local optimization
    for seed in range(20):
        if time.time() > deadline - 20.0:
            break
        
        if seed < 8:
            x0 = _tammes_init(seed)
        else:
            x0 = _random_init(seed)
        
        x0 = _enforce_constraints(x0)
        
        # Local optimization with L-BFGS-B
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x_opt = _enforce_constraints(result.x)
            score = -_objective(x_opt)
            
            if score > best_score:
                best_score = score
                best_points = x_opt.copy()
        except Exception:
            pass
    
    # Strategy 2: Differential evolution for global search
    if time.time() < deadline - 30.0:
        try:
            # Bounds: x in [0, 2], y in [0, sqrt(3)]
            bounds = [(0.0, 2.0), (0.0, np.sqrt(3.0))] * N
            
            # Use a smaller population for speed
            de_result = differential_evolution(
                _objective,
                bounds,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False,
                init='sobol'
            )
            
            x_de = _enforce_constraints(de_result.x)
            score_de = -_objective(x_de)
            
            if score_de > best_score:
                best_score = score_de
                best_points = x_de.copy()
                
                # Refine with L-BFGS-B
                if time.time() < deadline - 10.0:
                    try:
                        result2 = minimize(
                            _objective,
                            x_de,
                            method='L-BFGS-B',
                            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                        )
                        x_refined = _enforce_constraints(result2.x)
                        score_refined = -_objective(x_refined)
                        if score_refined > best_score:
                            best_score = score_refined
                            best_points = x_refined.copy()
                    except Exception:
                        pass
        except Exception:
            pass
    
    # Strategy 3: More local refinement from best point with perturbations
    if best_points is not None and time.time() < deadline - 10.0:
        rng = np.random.RandomState(123)
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            # Add small perturbation
            perturbation = rng.normal(0, 0.02, size=N * 2)
            x_pert = best_points + perturbation
            x_pert = _enforce_constraints(x_pert)
            
            try:
                result = minimize(
                    _objective,
                    x_pert,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                x_opt = _enforce_constraints(result.x)
                score = -_objective(x_opt)
                
                if score > best_score:
                    best_score = score
                    best_points = x_opt.copy()
            except Exception:
                pass
    
    # Final: ensure best_points is valid
    if best_points is None:
        # Fallback: use Tammes initialization
        best_points = _tammes_init(0)
        best_points = _enforce_constraints(best_points)
    
    best_points = best_points.reshape(N, 2)
    
    # Scale to unit-area triangle: divide by sqrt(3) to get unit area
    # The triangle (0,0), (2,0), (1,sqrt(3)) has area sqrt(3)
    # To get unit area, scale by 1/sqrt(3)
    result = best_points / np.sqrt(3.0)
    
    return result
