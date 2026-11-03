import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def compute_covering_radius(points):
    # Normalize points to ensure they are on the unit sphere
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # If hull computation fails, return a large value
        return np.pi
    
    faces = hull.simplices
    voronoi_vertices = []
    
    for face in faces:
        a = points[face[0]]
        b = points[face[1]]
        c = points[face[2]]
        
        # Compute normal to the plane
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        
        # Distance from origin to the plane
        d = np.dot(n, a)
        
        # Circumcenter
        t = d / (n_norm ** 2)
        o = t * n
        
        # Normalize to get point on sphere
        o_norm = np.linalg.norm(o)
        if o_norm < 1e-10:
            continue
        o = o / o_norm
        
        voronoi_vertices.append(o)
    
    if not voronoi_vertices:
        return np.pi
        
    voronoi_vertices = np.array(voronoi_vertices)
    
    # For each point, find the nearest Voronoi vertex and compute the angular distance
    # Use a simple loop for clarity, can be vectorized if needed
    max_dist = 0.0
    for p in points:
        # Compute angular distance to all Voronoi vertices
        # cos(theta) = p . v
        dots = np.dot(voronoi_vertices, p)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dist = np.min(dists)
        if min_dist > max_dist:
            max_dist = min_dist
            
    return max_dist

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = compute_covering_radius(best_points)
    
    # Define the objective function for the optimizer
    def objective(x):
        if time.time() > deadline:
            return best_radius  # Return best known if time is up
        pts = x.reshape(N, 3)
        # Normalize points
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        radius = compute_covering_radius(pts)
        return radius
    
    # Use Nelder-Mead for optimization
    # This is a derivative-free method, suitable for non-smooth functions
    # We optimize 390 parameters (3 per point)
    
    # To speed up, we can try multiple random restarts or a better initial guess
    # But let's start with the Fibonacci spiral
    
    # Nelder-Mead can be slow for high dimensions. We might need to reduce the dimensionality
    # or use a different optimizer.
    
    # Let's try a simple local search with random perturbations first, then refine with Nelder-Mead
    # Or use L-BFGS-B with a smooth approximation.
    
    # For now, let's use Nelder-Mead with a limited number of iterations
    # and check the time.
    
    # Since Nelder-Mead is slow for 390 dimensions, let's try a different approach:
    # Minimize the sum of the squares of the distances to the Voronoi vertices,
    # or use a smooth max approximation.
    
    # Let's use a smooth max approximation: soft-max with a high temperature
    def objective_smooth(x):
        if time.time() > deadline:
            return best_radius
        pts = x.reshape(N, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        
        # Compute covering radius with smooth max
        try:
            hull = ConvexHull(pts)
        except Exception:
            return np.pi
        
        faces = hull.simplices
        voronoi_vertices = []
        
        for face in faces:
            a = pts[face[0]]
            b = pts[face[1]]
            c = pts[face[2]]
            
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            
            d = np.dot(n, a)
            t = d / (n_norm ** 2)
            o = t * n
            o_norm = np.linalg.norm(o)
            if o_norm < 1e-10:
                continue
            o = o / o_norm
            voronoi_vertices.append(o)
        
        if not voronoi_vertices:
            return np.pi
            
        voronoi_vertices = np.array(voronoi_vertices)
        
        # Compute distances
        # For each point, find the minimum distance to a Voronoi vertex
        # Then take the maximum over all points
        # Use soft-max to approximate the max
        max_dist = 0.0
        for p in pts:
            dots = np.dot(voronoi_vertices, p)
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots)
            # Soft max: -log(sum(exp(-beta * dists))) / beta
            # For large beta, this approximates the max
            beta = 10.0
            soft_max = -np.log(np.sum(np.exp(-beta * dists))) / beta
            if soft_max > max_dist:
                max_dist = soft_max
                
        return max_dist

    # Use L-BFGS-B for optimization
    # This is faster than Nelder-Mead for high dimensions
    res = minimize(objective_smooth, points.flatten(), method='L-BFGS-B', options={'maxiter': 1000, 'maxfun': 10000})
    
    if res.success and res.fun < best_radius:
        best_points = res.x.reshape(N, 3)
        best_radius = res.fun
        # Refine with the exact objective
        best_radius = compute_covering_radius(best_points)
    
    # Check time and do a final polish if possible
    if time.time() < deadline - 10:
        # Try a few random restarts or local refinements
        for _ in range(5):
            if time.time() > deadline:
                break
            # Perturb the best points slightly
            noise = np.random.normal(0, 0.01, (N, 3))
            new_points = best_points + noise
            new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
            
            res2 = minimize(objective_smooth, new_points.flatten(), method='L-BFGS-B', options={'maxiter': 500, 'maxfun': 5000})
            if res2.success and res2.fun < best_radius:
                best_points = res2.x.reshape(N, 3)
                best_radius = compute_covering_radius(best_points)
                
    return best_points
