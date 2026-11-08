"""
Heilbronn triangles: maximize min triangle area for 11 points in unit-area equilateral triangle.
Strategy: Vectorized objective + Nelder-Mead local optimization + Structured Grid Initialization.
"""
import numpy as np
import random
import time
from scipy.optimize import minimize

N = 11

# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
AREA = np.sqrt(3) / 4.0  # area of the equilateral triangle

def _min_triangle_area_vectorized(points):
    """
    Compute the minimum area of any triangle formed by 3 of the points.
    Returns raw area (not normalized).
    Vectorized for speed.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    # Create indices for all triples i < j < k
    # For N=11, C(11,3) = 165. This is small.
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p_i = points[i_flat]
    p_j = points[j_flat]
    p_k = points[k_flat]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    if len(areas) == 0:
        return 0.0
        
    return np.min(areas)

def _is_inside_triangle(p):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -1e-9:
        return False
    if x < -1e-9 or x > 1.0 + 1e-9:
        return False
    max_y = np.sqrt(3) * x if x <= 0.5 else np.sqrt(3) * (1.0 - x)
    return y <= max_y + 1e-9

def _is_inside_triangle_batch(pts):
    """Vectorized check for multiple points."""
    x = pts[:, 0]
    y = pts[:, 1]
    mask = (y >= -1e-9) & (x >= -1e-9) & (x <= 1.0 + 1e-9)
    max_y = np.where(x <= 0.5, np.sqrt(3) * x, np.sqrt(3) * (1.0 - x))
    mask &= (y <= max_y + 1e-9)
    return mask

def _generate_structured_initial(rng, seed=0):
    """
    Generate structured initial configurations.
    Uses a base of vertices, midpoints, centroid, and centers of sub-regions.
    """
    points = []
    
    # 1. Vertices
    points.append(V0.copy())
    points.append(V1.copy())
    points.append(V2.copy())
    
    # 2. Midpoints
    points.append((V0 + V1) / 2.0)
    points.append((V1 + V2) / 2.0)
    points.append((V2 + V0) / 2.0)
    
    # 3. Centroid
    c = (V0 + V1 + V2) / 3.0
    points.append(c)
    
    # 4. Centers of the 4 regions formed by connecting centroid to vertices
    m01 = (V0 + V1) / 2.0
    m12 = (V1 + V2) / 2.0
    m20 = (V2 + V0) / 2.0
    
    # Region 1: V0, m01, c
    p1 = (V0 + m01 + c) / 3.0
    # Region 2: V1, m12, c
    p2 = (V1 + m12 + c) / 3.0
    # Region 3: V2, m20, c
    p3 = (V2 + m20 + c) / 3.0
    # Region 4: m01, m12, m20 (central medial triangle)
    p4 = (m01 + m12 + m20) / 3.0
    
    points.append(p1)
    points.append(p2)
    points.append(p3)
    points.append(p4)
    
    # We have 11 points now.
    # Add small random jitter to break exact symmetries
    points = np.array(points)
    points += rng.uniform(-0.01, 0.01, size=points.shape)
    
    # Project back inside if jitter pushed them out
    for i in range(len(points)):
        if not _is_inside_triangle(points[i]):
             points[i] = points[i] * 0.9 + c * 0.1
             
    return points

def _objective(params):
    """
    Objective function for Nelder-Mead.
    Returns the negative of the min triangle area (to be minimized).
    """
    points = params.reshape(-1, 2)
    
    # Check validity
    if not np.all(_is_inside_triangle_batch(points)):
        return 0.0
    
    min_area = _min_triangle_area_vectorized(points)
    return -min_area

def _optimize_nelder_mead(points, max_iter=500, tol=1e-6, time_limit=10.0):
    """
    Run Nelder-Mead optimization on the point configuration.
    """
    start_time = time.time()
    initial_params = points.flatten()
    
    try:
        res = minimize(
            _objective,
            initial_params,
            method='Nelder-Mead',
            options={
                'maxiter': max_iter,
                'xatol': tol,
                'fatol': tol,
                'disp': False
            }
        )
        
        best_params = res.x.reshape(-1, 2)
        
        # Ensure final validity
        if not np.all(_is_inside_triangle_batch(best_params)):
             return points, _min_triangle_area_vectorized(points)
            
        return best_params, _min_triangle_area_vectorized(best_params)
    
    except Exception:
        return points, _min_triangle_area_vectorized(points)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    best_points = None
    best_min = 0.0
    
    rng = random.Random(42)
    
    # Strategy 1: Structured initialization + Nelder-Mead
    n_structured_tries = 10
    
    for i in range(n_structured_tries):
        if time.time() > deadline:
            break
            
        init_points = _generate_structured_initial(rng, seed=i)
        opt_points, min_area = _optimize_nelder_mead(init_points, max_iter=500, time_limit=10.0)
        
        if min_area > best_min:
            best_min = min_area
            best_points = opt_points.copy()
            
    # Strategy 2: Random restarts + Nelder-Mead
    n_random_tries = 5
    for i in range(n_random_tries):
        if time.time() > deadline - 20:
            break
            
        points = []
        while len(points) < N:
            u = rng.random()
            v = rng.random()
            if u + v <= 1.0:
                p = V0 * (1 - u - v) + V1 * u + V2 * v
                points.append(p)
        points = np.array(points)
        
        opt_points, min_area = _optimize_nelder_mead(points, max_iter=300, time_limit=15.0)
        
        if min_area > best_min:
            best_min = min_area
            best_points = opt_points.copy()
            
    # Strategy 3: Final polish
    if best_points is not None and time.time() < deadline - 5:
        final_points, final_min = _optimize_nelder_mead(best_points, max_iter=200, tol=1e-8, time_limit=10.0)
        if final_min > best_min:
            best_min = final_min
            best_points = final_points.copy()

    if best_points is None:
        best_points = _generate_structured_initial(rng, seed=0)
    
    return best_points
# EVOLVE-BLOCK-END
