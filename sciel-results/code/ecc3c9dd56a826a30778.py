import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms

def _covering_radius(points):
    """
    Compute the exact covering radius of the points on the unit sphere.
    Returns the radius in radians, the indices of the worst facet, 
    and the circumcenter of that facet.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback if hull fails (should be rare with random perturbations)
        # Use a coarse grid estimate
        n_theta = 50
        n_phi = 100
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        theta_grid = theta_grid.flatten()
        phi_grid = phi_grid.flatten()
        
        grid_points = np.column_stack([
            np.sin(theta_grid) * np.cos(phi_grid),
            np.sin(theta_grid) * np.sin(phi_grid),
            np.cos(theta_grid)
        ])
        dots = grid_points @ points.T
        dots = np.clip(dots, -1.0, 1.0)
        max_dots = np.max(dots, axis=1)
        min_max_dot = np.min(max_dots)
        radius = np.arccos(min_max_dot)
        return radius, np.array([0, 1, 2]), grid_points[np.argmin(max_dots)]

    worst_radius = 0
    worst_simplex = None
    worst_circumcenter = None

    # Iterate over facets (triangles) of the convex hull
    for simplex in hull.simplices:
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        # Compute circumcenter of the spherical triangle
        # Normal to the plane defined by a, b, c
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        n = n / norm_n
        
        # The circumcenter is either n or -n.
        # We choose the one that is "close" to the triangle (dot product > 0 with a)
        if np.dot(n, a) < 0:
            n = -n
            
        # Distance from circumcenter to vertices
        r = np.arccos(np.clip(np.dot(n, a), -1, 1))
        
        if r > worst_radius:
            worst_radius = r
            worst_simplex = simplex
            worst_circumcenter = n

    if worst_simplex is None:
        # Degenerate case, return 0
        return 0.0, np.array([0, 1, 2]), points[0]
        
    return worst_radius, worst_simplex, worst_circumcenter

def _local_search(points, max_steps=200, step_size=0.1):
    """
    Local search to minimize the covering radius.
    Moves the vertices of the worst facet towards its circumcenter.
    """
    current_points = points.copy()
    best_radius, _, _ = _covering_radius(current_points)
    best_points = current_points.copy()
    
    for step in range(max_steps):
        radius, simplex, circumcenter = _covering_radius(current_points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = current_points.copy()
            
        # If radius is very small or we haven't improved in a while, we might be stuck
        # But for now, just keep moving towards the worst facet's center
        
        # Get the 3 points of the worst facet
        idx1, idx2, idx3 = simplex
        
        p1 = current_points[idx1]
        p2 = current_points[idx2]
        p3 = current_points[idx3]
        
        # Move towards circumcenter
        # We use a step size. To avoid oscillation, we might dampen the step.
        alpha = step_size
        
        # new_p = normalize(p + alpha * (circumcenter - p))
        # This is a gradient step on the distance to the circumcenter
        
        new_p1 = p1 + alpha * (circumcenter - p1)
        new_p2 = p2 + alpha * (circumcenter - p2)
        new_p3 = p3 + alpha * (circumcenter - p3)
        
        # Normalize
        new_p1 = new_p1 / np.linalg.norm(new_p1)
        new_p2 = new_p2 / np.linalg.norm(new_p2)
        new_p3 = new_p3 / np.linalg.norm(new_p3)
        
        current_points[idx1] = new_p1
        current_points[idx2] = new_p2
        current_points[idx3] = new_p3
        
        # Occasionally perturb other points to escape local minima?
        # For simplicity and speed, we stick to the greedy move on the worst facet.
        # The multi-start strategy handles the global search aspect.
        
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety

    best_points = None
    best_radius = np.inf

    # Number of restarts
    n_restarts = 50
    
    # Initial guess: Fibonacci sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    base_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    base_points = _normalize(base_points)
    
    # First, run local search on the base points to get a good starting point
    pts, rad = _local_search(base_points, max_steps=100, step_size=0.1)
    best_points = pts
    best_radius = rad

    # Multi-start local search
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
            
        rng = np.random.default_rng(seed=42 + restart)
        
        # Perturb the best points found so far
        # Use a small noise to explore nearby configurations
        noise_scale = 0.05 + 0.05 * (restart / n_restarts)
        noise = rng.normal(0, noise_scale, size=(N, 3))
        points = best_points + noise
        points = _normalize(points)
        
        # Run local search
        pts, rad = _local_search(points, max_steps=100, step_size=0.1)
        
        if rad < best_radius:
            best_radius = rad
            best_points = pts
            
        # If we are getting close to the theoretical limit, we can stop early
        # Best known is ~18.3 deg = 0.319 rad
        if best_radius < 0.315:
            # Try a few more with smaller noise to polish
            for polish in range(5):
                if time.time() > deadline:
                    break
                noise = rng.normal(0, 0.01, size=(N, 3))
                points = best_points + noise
                points = _normalize(points)
                pts, rad = _local_search(points, max_steps=50, step_size=0.05)
                if rad < best_radius:
                    best_radius = rad
                    best_points = pts

    # Final polish on the best point found
    if time.time() < deadline - 5.0:
        pts, rad = _local_search(best_points, max_steps=200, step_size=0.05)
        if rad < best_radius:
            best_radius = rad
            best_points = pts

    return best_points
