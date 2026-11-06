"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

DEADLINE = 145.0  # 5s safety margin before 150s kill


def _covering_radius_rad(points):
    """Compute covering radius in radians by sampling the sphere."""
    # Sample points on sphere to estimate max min-distance
    # Use a dense enough grid for accuracy
    n_theta = 100
    n_phi = 200
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi)
    THETA, PHI = np.meshgrid(thetas, phis)
    # Convert to Cartesian
    X = np.sin(THETA) * np.cos(PHI)
    Y = np.sin(THETA) * np.sin(PHI)
    Z = np.cos(THETA)
    samples = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Normalize points
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # For each sample, find min distance to any point
    # Use chunked computation to save memory
    max_min_dist = 0.0
    chunk_size = 5000
    for start in range(0, len(samples), chunk_size):
        chunk = samples[start:start + chunk_size]
        # Dot products
        dots = np.clip(chunk @ points.T, -1.0, 1.0)
        # Angular distances
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        max_min_dist = max(max_min_dist, np.max(min_dists))
    return max_min_dist


def _fibonacci_sphere(n):
    """Generate points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate random points on sphere."""
    vecs = rng.standard_normal((n, 3))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _repulsion_optimize(points, iterations=200, lr=0.01, rng=None):
    """Optimize points using repulsion forces to minimize covering radius."""
    if rng is None:
        rng = np.random.default_rng()
    points = points.copy()
    n = len(points)
    
    best_points = points.copy()
    best_radius = _covering_radius_rad(points)
    
    for it in range(iterations):
        # Compute pairwise distances
        dots = np.clip(points @ points.T, -1.0, 1.0)
        dists = np.arccos(dots)
        np.fill_diagonal(dists, np.pi)  # Self-distance
        
        # For each point, compute the "pressure" from covering radius perspective
        # Simple approach: each point is pushed away from its nearest neighbors
        # and pulled toward under-covered regions
        
        # Compute for each point the minimum distance to other points
        min_dists = np.min(dists, axis=1)
        
        # Gradient: move each point in direction away from nearest neighbor
        # Find nearest neighbor for each point
        nn_idx = np.argmin(dists, axis=1)
        
        # Compute repulsion force
        forces = np.zeros_like(points)
        for i in range(n):
            nn = nn_idx[i]
            if i == nn:
                continue
            # Direction from nn to i
            direction = points[i] - points[nn]
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-10:
                direction = direction / norm_dir
            # Project onto tangent plane
            tangent = direction - np.dot(direction, points[i]) * points[i]
            forces[i] += tangent / (min_dists[i] + 1e-6)
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        forces = forces / (force_norms + 1e-10)
        
        # Update points
        new_points = points + lr * forces
        # Normalize back to sphere
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        # Check if better
        new_radius = _covering_radius_rad(new_points)
        if new_radius < best_radius:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points
        else:
            # Reduce learning rate if stuck
            lr *= 0.95
    
    return best_points, best_radius


def construct():
    start_time = time.time()
    
    # Start with Fibonacci spiral
    points = _fibonacci_sphere(N)
    best_points = points.copy()
    best_radius = _covering_radius_rad(points)
    
    rng = np.random.default_rng(42)
    
    # Try multiple random restarts with optimization
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() > DEADLINE - 10:
            break
        
        # Initialize
        if restart == 0:
            init_points = _fibonacci_sphere(N)
        elif restart % 2 == 0:
            init_points = _random_sphere(N, rng)
        else:
            # Perturbed Fibonacci
            init_points = _fibonacci_sphere(N)
            noise = rng.standard_normal((N, 3)) * 0.1
            init_points = init_points + noise
            init_points = init_points / np.linalg.norm(init_points, axis=1, keepdims=True)
        
        # Optimize
        opt_points, opt_radius = _repulsion_optimize(
            init_points, iterations=100, lr=0.02, rng=rng
        )
        
        if opt_radius < best_radius:
            best_radius = opt_radius
            best_points = opt_points.copy()
    
    # Final check
    if time.time() > DEADLINE - 2:
        return best_points
    
    return best_points
# EVOLVE-BLOCK-END
