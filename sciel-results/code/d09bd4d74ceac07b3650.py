import numpy as np
from scipy.optimize import minimize
import time

N = 14

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    # Since points are generated from support functions, they are convex and ordered.
    # We can compute area directly as sum of cross products of consecutive vertices.
    # But to be safe and generic if called elsewhere, we can use shoelace formula.
    x = points[:, 0]
    y = points[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if area <= 1e-9:
        return 0.0

    # Calculate Minimum Triangle Area
    # Precompute triple indices for speed? N=14 is small (364 triples).
    min_area = float('inf')
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area = 0.5 * |cross(p2-p1, p3-p1)|
                v1 = p2 - p1
                v2 = p3 - p1
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                a = 0.5 * abs(cross)
                if a < min_area:
                    min_area = a
                    if min_area < 1e-9:
                        return 0.0
    return min_area / area

def _get_vertices_from_supports(h, angles):
    """
    Compute vertices of convex polygon given support distances h and normal angles.
    h: (N,) array
    angles: (N,) array of normal angles
    Returns: (N, 2) array of vertices
    """
    N = len(h)
    vertices = np.zeros((N, 2))
    
    # Intersection of line i-1 and line i
    # n_{i-1} . x = h_{i-1}
    # n_i . x = h_i
    # n = (cos a, sin a)
    # Matrix M = [[cos a_{i-1}, sin a_{i-1}], [cos a_i, sin a_i]]
    # b = [h_{i-1}, h_i]
    
    for i in range(N):
        a_prev = angles[(i-1) % N]
        a_curr = angles[i]
        
        c1 = np.cos(a_prev)
        s1 = np.sin(a_prev)
        c2 = np.cos(a_curr)
        s2 = np.sin(a_curr)
        
        det = c1 * s2 - s1 * c2
        
        if abs(det) < 1e-9:
            # Degenerate, shouldn't happen with distinct angles
            vertices[i] = np.array([0, 0])
            continue
            
        # x = (h1*s2 - h2*s1) / det
        # y = (c1*h2 - c2*h1) / det
        x = (h[(i-1)%N] * s2 - h[i] * s1) / det
        y = (c1 * h[i] - c2 * h[(i-1)%N]) / det
        
        vertices[i] = np.array([x, y])
        
    return vertices

def _objective_supports(h):
    """
    Objective function for support function optimization.
    Minimizes negative score.
    """
    if np.any(h <= 0):
        return 1e6 # Penalty for non-convex/invalid
        
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    vertices = _get_vertices_from_supports(h, angles)
    
    # Check if vertices are in correct order and form a simple convex polygon
    # The construction guarantees convexity if h > 0 and angles are increasing.
    
    score = _calculate_score(vertices)
    return -score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial guess: Regular 14-gon
    # For a regular polygon with apothem a, h_i = a for all i.
    # Let's start with h = ones.
    h0 = np.ones(N)
    
    best_score = 0.0
    best_h = h0.copy()
    
    # Optimization
    # We use L-BFGS-B as it handles bounds well.
    # h_i must be > 0.
    bounds = [(0.1, 10.0)] * N
    
    try:
        res = minimize(
            _objective_supports,
            h0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if -res.fun > best_score:
            best_score = -res.fun
            best_h = res.x
            
    except Exception:
        pass
        
    # Try a few different initializations to avoid local minima
    # 1. Slightly perturbed regular
    # 2. "Jittered" supports
    
    inits = [
        np.ones(N),
        1.0 + 0.1 * np.random.randn(N),
        1.0 + 0.2 * np.sin(np.linspace(0, 2*np.pi, N, endpoint=False))
    ]
    
    for h_init in inits:
        if time.time() > deadline - 20:
            break
        h_init = np.clip(h_init, 0.1, 10.0)
        
        try:
            res = minimize(
                _objective_supports,
                h_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if -res.fun > best_score:
                best_score = -res.fun
                best_h = res.x
                
        except Exception:
            pass
            
    # Final polish with Nelder-Mead on the best found
    if time.time() < deadline - 5:
        try:
            res_nm = minimize(
                _objective_supports,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_h = res_nm.x
        except Exception:
            pass
        
    # Convert best support distances to points
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    best_points = _get_vertices_from_supports(best_h, angles)
    
    # Verify score
    final_score = _calculate_score(best_points)
    if final_score <= 0:
        # Fallback to regular 14-gon
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
