import numpy as np
from scipy.optimize import minimize
import time

N = 130

def _cartesian_to_spherical(points):
    """
    Convert Cartesian points (N, 3) to spherical coordinates (N, 2).
    Returns theta (polar, 0 to pi) and phi (azimuthal, 0 to 2pi).
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    # Avoid division by zero, though r should be 1
    r_safe = np.where(r == 0, 1e-9, r)
    
    theta = np.arccos(np.clip(z / r_safe, -1.0, 1.0))
    phi = np.arctan2(y, x)
    
    # Ensure phi is in [0, 2pi)
    phi = phi % (2 * np.pi)
    
    return np.column_stack([theta, phi])

def _spherical_to_cartesian(coords):
    """
    Convert spherical coordinates (N, 2) to Cartesian points (N, 3).
    """
    theta = coords[:, 0]
    phi = coords[:, 1]
    
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    return np.column_stack([x, y, z])

def _covering_radius(points, n_grid=20000):
    """
    Estimate the covering radius by sampling the sphere.
    """
    # Generate a dense grid on the sphere
    # Using a Fibonacci spiral for the grid is efficient and uniform
    i = np.arange(n_grid) + 0.5
    z_grid = 1.0 - 2.0 * i / n_grid
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i
    r_grid = np.sqrt(1.0 - z_grid * z_grid)
    grid_points = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    # Calculate distance from each grid point to the nearest generator
    # To save memory/time, we can chunk this or use cdist
    # cdist is efficient for moderate sizes
    from scipy.spatial.distance import cdist
    
    # We only need the minimum distance for each grid point
    # dists shape: (n_grid, N)
    dists = cdist(grid_points, points)
    min_dists = np.min(dists, axis=1)
    
    # Convert Euclidean distance to angular distance
    # d = 2 * sin(theta/2)  => theta = 2 * arcsin(d/2)
    # Clip to avoid numerical issues in arcsin
    min_dists_clipped = np.clip(min_dists, 0.0, 2.0)
    angular_dists = 2.0 * np.arcsin(min_dists_clipped / 2.0)
    
    return np.max(angular_dists)

def _repulsive_energy(coords, lambda_param=50.0):
    """
    Calculate the repulsive energy for points defined by spherical coordinates.
    Energy = sum_{i<j} exp(-lambda * ||P_i - P_j||^2)
    Minimizing this maximizes the separation.
    """
    points = _spherical_to_cartesian(coords)
    
    # Calculate pairwise squared distances
    # Use the identity ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    # Since points are on unit sphere, ||a||^2 = 1, ||b||^2 = 1
    # ||a - b||^2 = 2 - 2 a.b
    dot_products = points @ points.T
    sq_dists = 2.0 - 2.0 * dot_products
    
    # Ensure numerical stability (sq_dists should be >= 0)
    sq_dists = np.clip(sq_dists, 0.0, 4.0)
    
    # Mask the diagonal (i == j) to avoid self-interaction
    mask = ~np.eye(N, dtype=bool)
    sq_dists_masked = sq_dists[mask]
    
    # Calculate energy
    energy = np.sum(np.exp(-lambda_param * sq_dists_masked))
    
    return energy

def _optimize_single(start_coords, time_limit=10.0):
    """
    Optimize a single configuration starting from start_coords.
    """
    start_time = time.time()
    
    # Bounds for theta: [0, pi], phi: [0, 2pi]
    bounds = [(0.0, np.pi), (0.0, 2.0 * np.pi)] * N
    
    # Flatten coordinates for the optimizer
    x0 = start_coords.flatten()
    
    # Objective function wrapper
    def objective(x_flat):
        if time.time() - start_time > time_limit:
            # Return a large value to signal stopping, though L-BFGS-B doesn't handle this gracefully
            # It's better to just let it run and check time outside, or use a maxiter that fits in time
            pass
        coords = x_flat.reshape(N, 2)
        return _repulsive_energy(coords)
        
    # We don't have an easy gradient for the spherical coords with exp kernel, 
    # so we rely on L-BFGS-B's numerical gradient or provide a simple one?
    # Numerical gradient is slow. Let's provide the gradient?
    # Actually, for N=130, numerical gradient might be too slow for many iterations.
    # However, L-BFGS-B is quite efficient. 
    # Let's try without providing gradient first, but limit iterations.
    
    try:
        res = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False}
        )
    except Exception:
        return start_coords

    if res.success or res.status in [1, 2]: # 1: maxiter, 2: maxfev
        return res.x.reshape(N, 2)
    else:
        # If it failed, maybe return the best found so far? 
        # res.x is usually the last iterate.
        return res.x.reshape(N, 2)

def construct():
    # Deadline guard
    start_time = time.time()
    total_time_limit = 110.0 # Leave 10s margin for 120s limit
    
    best_score = np.inf
    best_points = None
    
    # Generate multiple starting points
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    spiral_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Random Uniform
    rng = np.random.default_rng(42)
    random_points = rng.normal(size=(N, 3))
    random_points /= np.linalg.norm(random_points, axis=1, keepdims=True)
    
    # 3. Icosahedron-based (High symmetry)
    # Generate 12 vertices of icosahedron and jitter
    t = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ])
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # We need 130 points. We can take the icosahedron and its dual (dodecahedron) or just jitter and duplicate?
    # Better: Just use the spiral and random restarts. The icosahedron only gives 12 points.
    # We can use the "Golden Spiral" as the primary good start.
    
    starts = [spiral_points, random_points]
    
    # Add a few more random restarts if time permits
    for i in range(5):
        if time.time() - start_time > total_time_limit - 20.0:
            break
        rp = rng.normal(size=(N, 3))
        rp /= np.linalg.norm(rp, axis=1, keepdims=True)
        starts.append(rp)
        
    for idx, start_pts in enumerate(starts):
        if time.time() - start_time > total_time_limit - 15.0:
            break
            
        # Convert to spherical
        start_coords = _cartesian_to_spherical(start_pts)
        
        # Optimize
        # Adjust time limit per optimization based on how many are left
        remaining = total_time_limit - (time.time() - start_time)
        n_left = len(starts) - idx
        time_per_opt = max(5.0, (remaining - 10.0) / n_left) # Keep 10s buffer
        
        opt_coords = _optimize_single(start_coords, time_limit=time_per_opt)
        
        # Convert back to Cartesian
        opt_points = _spherical_to_cartesian(opt_coords)
        
        # Normalize to ensure they are on the sphere (numerical drift)
        opt_points /= np.linalg.norm(opt_points, axis=1, keepdims=True)
        
        # Evaluate covering radius
        score = _covering_radius(opt_points)
        
        if score < best_score:
            best_score = score
            best_points = opt_points
            
        # Check if we found something very good
        if best_score < 0.197: # ~11.3 degrees
             # We could stop early, but let's see if we can improve further
             pass

    # Fallback if no optimization succeeded
    if best_points is None:
        best_points = spiral_points
        best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
        
    return best_points
