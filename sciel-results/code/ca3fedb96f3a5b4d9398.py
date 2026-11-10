import numpy as np
import time
from scipy.spatial import ConvexHull
import math

N = 130

def _compute_covering_radius_and_centers(points):
    """
    Computes the covering radius (max circumradius of hull facets) 
    and returns the radius, the list of (center, facet_indices) for facets 
    with radius close to the maximum, and the hull object.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0, [], None

    max_radius = 0.0
    facet_radii = []
    facet_centers = []
    
    for i, simplex in enumerate(hull.simplices):
        u = points[simplex[0]]
        v = points[simplex[1]]
        w = points[simplex[2]]
        
        uv = np.dot(u, v)
        uw = np.dot(u, w)
        vw = np.dot(v, w)
        
        M = np.array([
            [1.0, uv, uw],
            [uv, 1.0, vw],
            [uw, vw, 1.0]
        ])
        
        try:
            x = np.linalg.solve(M, np.ones(3))
        except np.linalg.LinAlgError:
            continue
            
        xT_M_x = np.dot(x, M @ x)
        
        if xT_M_x <= 1e-10:
            continue
            
        cos_R = 1.0 / np.sqrt(xT_M_x)
        cos_R = np.clip(cos_R, -1.0, 1.0)
        
        C = x[0] * u + x[1] * v + x[2] * w
        norm_C = np.linalg.norm(C)
        if norm_C < 1e-10:
            continue
        C = C / norm_C
        
        R = math.acos(cos_R)
        facet_radii.append(R)
        facet_centers.append(C)
        
        if R > max_radius:
            max_radius = R

    # Find facets within a small tolerance of the max radius
    # This allows the local search to address multiple "worst" facets
    tol = 1e-4
    active_facets = []
    for i, r in enumerate(facet_radii):
        if r >= max_radius - tol:
            active_facets.append((facet_centers[i], hull.simplices[i]))

    return math.degrees(max_radius), active_facets, hull

def _move_point_towards(p, c, step_size):
    """Move point p on the sphere towards c by step_size radians."""
    dot_pc = np.dot(p, c)
    # Tangent vector from p to c
    t = c - dot_pc * p
    norm_t = np.linalg.norm(t)
    if norm_t < 1e-10:
        return p
    t = t / norm_t
    new_p = p + step_size * t
    new_p = new_p / np.linalg.norm(new_p)
    return new_p

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # 1. Initialization: Fibonacci spiral
    np.random.seed(42)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Small jitter
    noise = np.random.randn(N, 3) * 0.01
    points = points + noise
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    best_points = points.copy()
    best_radius, _, _ = _compute_covering_radius_and_centers(best_points)
    
    # 2. Iterated Local Search
    alpha = 0.05
    max_iter_per_restart = 500
    
    while time.time() < deadline:
        # Local Search Phase
        for _ in range(max_iter_per_restart):
            if time.time() > deadline:
                break
                
            radius, active_facets, hull = _compute_covering_radius_and_centers(points)
            
            if radius < best_radius:
                best_radius = radius
                best_points = points.copy()
                
            if not active_facets:
                break
                
            # Pick a random active facet to modify
            # This helps escape symmetries and local minima
            center, simplex = active_facets[np.random.randint(len(active_facets))]
            
            improved = False
            # Try moving vertices of this facet towards its center
            # We try a few different step sizes
            for step in [alpha, alpha * 0.5, alpha * 2.0]:
                new_points = points.copy()
                for v_idx in simplex:
                    new_points[v_idx] = _move_point_towards(points[v_idx], center, step)
                
                new_radius, _, _ = _compute_covering_radius_and_centers(new_points)
                
                if new_radius < radius:
                    points = new_points
                    improved = True
                    break
            
            if not improved:
                # If we can't improve with the current alpha, reduce it
                alpha *= 0.9
                if alpha < 0.001:
                    break
        
        # Perturbation Phase (Kick)
        # If we are still within time, perturb the current best points
        if time.time() > deadline - 10:
            break
            
        # Perturb a random subset of points
        k = np.random.randint(5, 15) # Perturb 5-15 points
        indices = np.random.choice(N, k, replace=False)
        
        new_points = best_points.copy()
        for idx in indices:
            # Add noise
            noise = np.random.randn(3) * 0.1
            new_points[idx] = new_points[idx] + noise
            new_points[idx] = new_points[idx] / np.linalg.norm(new_points[idx])
        
        # Check if perturbation is valid (not degenerate)
        pert_radius, _, _ = _compute_covering_radius_and_centers(new_points)
        
        # Accept perturbation if it's not significantly worse, or always accept to explore?
        # Standard ILS: accept if better, or with some probability if worse.
        # Given the time constraint, let's accept if it's better than the current local min,
        # or just use it as the new starting point for the next local search.
        # Actually, restarting from the BEST point with a kick is safer.
        
        points = new_points
        alpha = 0.05 # Reset step size for new local search

    return best_points
