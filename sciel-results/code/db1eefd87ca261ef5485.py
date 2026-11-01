"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # First, get a good initial packing using hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            points = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98
    
    # Compute initial score
    def compute_score(pts):
        # Check validity
        if len(pts) != N:
            return -1.0
        if not np.all(np.isfinite(pts)):
            return -1.0
        # Check inside container
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return -1.0
        # Check non-overlapping
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    min_dist = d
        r = min(min_dist / 2.0, 1.0 - np.max(dists))
        return r
    
    initial_score = compute_score(points)
    if initial_score <= 0:
        return points
    
    # Try to improve using local optimization
    # Use a subset of points to optimize for speed
    n_opt = min(N, 100)  # Optimize only first n_opt points for speed
    opt_indices = np.arange(n_opt)
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        pts = points.copy()
        pts[opt_indices] = params.reshape(-1, 2)
        
        # Check inside container
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return 1e10
        
        # Compute minimum distance between all pairs
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    min_dist = d
        
        # Score is the minimum of (min_dist/2, 1 - max_dist)
        return -(min(min_dist / 2.0, 1.0 - np.max(dists)))
    
    # Initial parameters
    x0 = points[opt_indices].flatten()
    
    # Bounds to keep points inside container
    bounds = [(-0.99, 0.99)] * (2 * n_opt)
    
    # Try to optimize
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 50, 'maxfun': 1000, 'ftol': 1e-8})
        if result.fun < 1e10:
            new_points = points.copy()
            new_points[opt_indices] = result.x.reshape(-1, 2)
            new_score = compute_score(new_points)
            if new_score > initial_score:
                points = new_points
    except Exception:
        pass
    
    # If we have time, try a more global approach with different ring configurations
    if time.time() < deadline - 30:
        # Try radial ring approach
        best_ring_points = None
        best_ring_score = initial_score
        
        # Try different numbers of rings
        for n_rings in range(5, 15):
            if time.time() > deadline - 10:
                break
            
            # Generate ring configuration
            ring_radii = np.linspace(0.1, 0.95, n_rings)
            ring_points = []
            
            for r_ring in ring_radii:
                # Number of circles that can fit on this ring
                # Circumference is 2*pi*r_ring, need spacing >= 2*circle_radius
                # Estimate circle radius from density
                est_r = 0.05
                n_circ = max(1, int(2 * np.pi * r_ring / (2 * est_r)))
                
                # Place circles on this ring
                angles = np.linspace(0, 2*np.pi, n_circ, endpoint=False)
                for a in angles:
                    x = r_ring * np.cos(a)
                    y = r_ring * np.sin(a)
                    ring_points.append([x, y])
            
            # Add center point
            ring_points.append([0.0, 0.0])
            
            ring_points = np.array(ring_points)
            
            # If we have too many points, subsample
            if len(ring_points) > N:
                # Keep points closest to center
                dists = np.linalg.norm(ring_points, axis=1)
                idx = np.argsort(dists)[:N]
                ring_points = ring_points[idx]
            elif len(ring_points) < N:
                continue
            
            score = compute_score(ring_points)
            if score > best_ring_score:
                best_ring_score = score
                best_ring_points = ring_points.copy()
        
        if best_ring_points is not None:
            points = best_ring_points
    
    # Final validation
    final_score = compute_score(points)
    if final_score <= 0:
        # Return the initial valid packing
        return points
    
    return points
# EVOLVE-BLOCK-END
