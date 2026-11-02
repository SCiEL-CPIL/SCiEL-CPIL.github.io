"""
Heilbronn triangles: maximize the smallest triangle area for 11 points.
Strategy: Structural Symmetry Reduction + Global/Local Hybrid Optimization.

1. Structural Assumption:
   The optimal configuration for 11 points in an equilateral triangle likely exhibits
   symmetry. We assume a configuration with 3-fold rotational symmetry ($C_3$) 
   or reflection symmetry. 
   A common candidate structure for N=11 is:
   - 3 vertices of the triangle.
   - 3 midpoints of the sides.
   - 3 points forming an inner equilateral triangle (concentric).
   - 2 points on the axis of symmetry (or 1 center + 1 other).
   However, N=11 is odd. A likely symmetric partition is:
   - 3 vertices (fixed).
   - 3 midpoints (fixed or variable along edges).
   - 3 points in an inner orbit (rotational symmetry).
   - 1 center point.
   - 1 additional point on an axis? 
   Actually, for N=11, a known heuristic structure is a triangular lattice perturbation.
   Let's try a more flexible symmetry: 
   Points are divided into orbits.
   Orbit 1: 3 vertices (fixed).
   Orbit 2: 3 points on edges (symmetric).
   Orbit 3: 3 points in interior (rotational symmetry).
   Orbit 4: 2 points on an axis of symmetry (reflection symmetry).
   Total: 3 + 3 + 3 + 2 = 11.
   
   This reduces variables from 22 to:
   - Orbit 2: 1 parameter (distance from vertex).
   - Orbit 3: 2 parameters (radius, angle offset from axis).
   - Orbit 4: 2 parameters (positions along axis).
   Total variables: 5. This is very low dimensional and easy to optimize.
   
   However, if the true optimum breaks this symmetry, we might miss it.
   To be safer, we will also try a "less symmetric" parameterization:
   - 3 vertices.
   - 3 midpoints.
   - 5 interior points optimized freely (10 vars) but initialized near symmetric positions.
   
   Let's implement a multi-strategy approach:
   1. High-Symmetry Model (5 vars): 3 verts, 3 edge-mid, 3 inner-rot, 2 axis.
   2. Medium-Symmetry Model (13 vars): 3 verts, 3 edge-mid, 5 free interior.
   3. Full Search (22 vars): All free, but initialized smartly.
   
   We will use Differential Evolution for the global search on the reduced/medium models
   and Nelder-Mead for polishing.
"""

import numpy as np
import time
import random
from scipy.optimize import differential_evolution, minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Pre-compute triplet indices
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    x = pts[:, 0]
    y = pts[:, 1]
    
    if np.any(y < -eps): return False
    if np.any(y > np.sqrt(3) * x + eps): return False
    if np.any(y > np.sqrt(3) * (1 - x) + eps): return False
    if np.any(x < -eps): return False
    if np.any(x > 1 + eps): return False
    
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    
    y = np.clip(y, 0, S3H)
    
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    
    x = np.clip(x, x_min, x_max)
    
    pts[:, 0] = x
    pts[:, 1] = y
    return pts

def _build_high_symmetry(pts_params):
    """
    Build 11 points from 5 parameters assuming high symmetry.
    Params:
    0: t_edge (0 to 0.5) - distance from vertex along edge for the 3 edge points.
       These are at (t,0), (1-t, t/sqrt(3)), (t/sqrt(3), 1-t/sqrt(3))? No, midpoints are specific.
       Let's define the 3 edge points as a fraction 'f' along the edge from vertex A to B, etc.
       Actually, for symmetry, let's place them at a fraction 'f' from the vertex.
       P1 (on AB): (f, 0)
       P2 (on BC): (1-f, f/sqrt(3) * ?) -> B is (1,0), C is (0.5, S3H).
       Vector BC = (-0.5, S3H). Point = B + f * (C-B) = (1 - 0.5f, f*S3H).
       P3 (on CA): C + f * (A-C) = (0.5 + f*(0-0.5), S3H + f*(0-S3H)) = (0.5(1-f), S3H(1-f)).
       
    1: r_inner - radius of the inner equilateral triangle (centered at centroid).
    2: theta_inner - angle offset of the inner triangle.
       Centroid is (0.5, S3H/3).
       Inner points:
       I1: (0.5 + r*cos(theta), S3H/3 + r*sin(theta))
       I2: (0.5 + r*cos(theta+2pi/3), S3H/3 + r*sin(theta+2pi/3))
       I3: (0.5 + r*cos(theta+4pi/3), S3H/3 + r*sin(theta+4pi/3))
       
    3: d_axis1 - distance from centroid along the vertical axis (upwards).
       A1: (0.5, S3H/3 + d_axis1)
    4: d_axis2 - distance from centroid along the vertical axis (downwards).
       A2: (0.5, S3H/3 - d_axis2)
       
    Vertices are fixed: (0,0), (1,0), (0.5, S3H).
    """
    f = pts_params[0]
    r = pts_params[1]
    theta = pts_params[2]
    d1 = pts_params[3]
    d2 = pts_params[4]
    
    pts = np.zeros((N, 2))
    
    # Vertices
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2] = [0.5, S3H]
    
    # Edge points
    # f is fraction from first vertex to second.
    # P3 on AB: (f, 0)
    pts[3] = [f, 0.0]
    # P4 on BC: B + f*(C-B)
    pts[4] = [1.0 - 0.5*f, S3H*f]
    # P5 on CA: C + f*(A-C)
    pts[5] = [0.5*(1.0-f), S3H*(1.0-f)]
    
    # Inner triangle
    cx, cy = 0.5, S3H/3
    angles = [theta, theta + 2*np.pi/3, theta + 4*np.pi/3]
    for i, ang in enumerate(angles):
        pts[6+i] = [cx + r*np.cos(ang), cy + r*np.sin(ang)]
        
    # Axis points
    pts[9] = [0.5, cy + d1]
    pts[10] = [0.5, cy - d2]
    
    return pts

def _build_medium_symmetry(pts_params):
    """
    Build 11 points from 13 parameters.
    0: f_edge (fraction for edge points)
    1-10: 5 free interior points (x, y) each.
    Wait, 3 verts + 3 edge + 5 interior = 11.
    Edge points are symmetric.
    Interior points are free.
    """
    f = pts_params[0]
    pts = np.zeros((N, 2))
    
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2] = [0.5, S3H]
    
    pts[3] = [f, 0.0]
    pts[4] = [1.0 - 0.5*f, S3H*f]
    pts[5] = [0.5*(1.0-f), S3H*(1.0-f)]
    
    # 5 free points
    for i in range(5):
        x = pts_params[1 + 2*i]
        y = pts_params[2 + 2*i]
        pts[6+i] = [x, y]
        
    return pts

def _objective_high_sym(params):
    pts = _build_high_symmetry(params)
    pts = _project_to_triangle(pts)
    if not _check_inside(pts):
        return 1e6 # Penalty
    return -_min_triangle_area(pts)

def _objective_medium_sym(params):
    pts = _build_medium_symmetry(params)
    pts = _project_to_triangle(pts)
    if not _check_inside(pts):
        return 1e6
    return -_min_triangle_area(pts)

def _objective_full(params):
    pts = params.reshape(N, 2)
    pts = _project_to_triangle(pts)
    if not _check_inside(pts):
        return 1e6
    return -_min_triangle_area(pts)

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    def _update_best(pts):
        nonlocal best_pts, best_score
        pts = _project_to_triangle(pts)
        if _check_inside(pts):
            score = _min_triangle_area(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
                return True
        return False

    # Strategy 1: High Symmetry Search (5 variables)
    # Bounds:
    # f: [0, 0.5] (midpoint is 0.5)
    # r: [0, 0.3] (radius)
    # theta: [0, 2pi]
    # d1: [0, 0.3]
    # d2: [0, 0.3]
    bounds_high = [(0.0, 0.5), (0.0, 0.3), (0.0, 2*np.pi), (0.0, 0.3), (0.0, 0.3)]
    
    if time.time() - start_time < DEADLINE - 20:
        try:
            res_high = differential_evolution(
                _objective_high_sym, 
                bounds_high, 
                seed=42, 
                maxiter=50, 
                popsize=15,
                tol=1e-8
            )
            if res_high.success or res_high.fun < 1e5:
                best_params_high = res_high.x
                _update_best(_build_high_symmetry(best_params_high))
                
                # Polish with Nelder-Mead
                res_nm = minimize(
                    _objective_high_sym, 
                    best_params_high, 
                    method='Nelder-Mead',
                    options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 2000}
                )
                _update_best(_build_high_symmetry(res_nm.x))
        except Exception:
            pass

    # Strategy 2: Medium Symmetry Search (13 variables)
    # f: [0, 0.5]
    # x, y for 5 points: [0, 1], [0, S3H]
    bounds_med = [(0.0, 0.5)]
    for _ in range(5):
        bounds_med.append((0.0, 1.0))
        bounds_med.append((0.0, S3H))
        
    if time.time() - start_time < DEADLINE - 20:
        try:
            # Use a smaller popsize and maxiter for this higher dim problem
            res_med = differential_evolution(
                _objective_medium_sym, 
                bounds_med, 
                seed=42, 
                maxiter=30, 
                popsize=10,
                tol=1e-6
            )
            if res_med.success or res_med.fun < 1e5:
                best_params_med = res_med.x
                _update_best(_build_medium_symmetry(best_params_med))
                
                res_nm = minimize(
                    _objective_medium_sym, 
                    best_params_med, 
                    method='Nelder-Mead',
                    options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 2000}
                )
                _update_best(_build_medium_symmetry(res_nm.x))
        except Exception:
            pass

    # Strategy 3: Full Free Search (22 variables) - Multi-start Nelder-Mead
    # Initialize with best known structures + random perturbations
    # 1. High sym best
    # 2. Medium sym best
    # 3. Random
    
    initializations = []
    
    if best_pts is not None:
        initializations.append(best_pts)
        
    # Add a few random starts
    for _ in range(5):
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1:
                u = rng.random()
                v = rng.random()
            x = u + 0.5 * v
            y = S3H * v
            pts[i] = [x, y]
        initializations.append(pts)
        
    # Also try a lattice-based init
    # 3x4 grid approx
    lattice_pts = []
    for i in range(4):
        for j in range(3):
            if len(lattice_pts) < N:
                # Map grid to triangle
                x = i / 3.0
                y = j / 2.0 * S3H * (1 - x) # Simplified, will be projected
                lattice_pts.append([x, y])
    if len(lattice_pts) == N:
        initializations.append(np.array(lattice_pts))

    for init_pts in initializations:
        if time.time() - start_time > DEADLINE - 5:
            break
        
        try:
            res_full = minimize(
                _objective_full, 
                init_pts.flatten(), 
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000}
            )
            _update_best(res_full.x.reshape(N, 2))
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        best_pts = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
            [0.5, 0.0], [0.25, S3H/2], [0.75, S3H/2],
            [0.5, S3H/3],
            [0.25, 0.0], [0.75, 0.0],
            [0.125, S3H/4], [0.875, S3H/4]
        ])
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")
