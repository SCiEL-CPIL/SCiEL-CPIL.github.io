"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time


def construct():
    """
    Pack 50 equal circles in a unit circle using ring-based parameterization
    and differential evolution optimization.
    
    Strategy:
    - Represent centers as concentric rings of regular polygons
    - Free variables: ring radii and rotational angles
    - Use differential evolution to find good ring configurations
    - Binary search on radius with soft constraint violation
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Ring configuration: number of circles per ring
    # We'll try several configurations and pick the best
    ring_configs = [
        [1, 6, 12, 18, 13],   # 50 total
        [1, 6, 12, 20, 11],
        [1, 6, 12, 16, 15],
        [1, 6, 10, 16, 17],
        [1, 6, 12, 14, 17],
        [1, 6, 10, 14, 19],
        [1, 6, 12, 12, 19],
        [1, 6, 8, 16, 19],
        [1, 6, 10, 12, 21],
        [1, 6, 12, 10, 21],
    ]
    
    best_points = None
    best_radius = 0.0
    
    def compute_centers(ring_params, ring_sizes):
        """Compute circle centers from ring parameters.
        
        ring_params: array of shape (2*n_rings,) where:
            [0, 1] = radius, angle for ring 0
            [2, 3] = radius, angle for ring 1
            etc.
        ring_sizes: list of number of circles in each ring
        """
        n_rings = len(ring_sizes)
        centers = []
        for ring_idx, n_circles in enumerate(ring_sizes):
            if n_circles == 0:
                continue
            ring_radius = ring_params[2 * ring_idx]
            ring_angle = ring_params[2 * ring_idx + 1]
            
            if n_circles == 1:
                # Single circle at center
                centers.append([0.0, 0.0])
            else:
                angles = ring_angle + 2.0 * np.pi * np.arange(n_circles) / n_circles
                xs = ring_radius * np.cos(angles)
                ys = ring_radius * np.sin(angles)
                for x, y in zip(xs, ys):
                    centers.append([x, y])
        
        return np.array(centers)
    
    def objective(ring_params, ring_sizes):
        """Objective function: maximize minimum distance (negative for minimization).
        
        Returns a score where lower is better (negative of min distance/2).
        """
        centers = compute_centers(ring_params, ring_sizes)
        
        # Check if we have exactly N points
        if len(centers) != N:
            return 1e6
        
        # Compute minimum distance between any pair
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2)
                if d < min_dist:
                    min_dist = d
        
        # Check if all circles are inside unit circle
        max_r = np.max(np.sqrt(centers[:, 0]**2 + centers[:, 1]**2))
        if max_r > 1.0:
            # Penalty for being outside
            return 1e6 + (max_r - 1.0)
        
        # Return negative half of min distance (we want to maximize min_dist)
        return -min_dist / 2.0
    
    def get_bounds(ring_sizes):
        """Get bounds for differential evolution."""
        n_rings = len(ring_sizes)
        bounds = []
        for ring_idx, n_circles in enumerate(ring_sizes):
            # Radius bound: 0 to 1
            bounds.append((0.0, 1.0))
            # Angle bound: 0 to 2*pi
            bounds.append((0.0, 2.0 * np.pi))
        return bounds
    
    for ring_sizes in ring_configs:
        if sum(ring_sizes) != N:
            continue
        
        if time.time() > deadline:
            break
        
        bounds = get_bounds(ring_sizes)
        
        # Run differential evolution
        try:
            result = differential_evolution(
                objective,
                bounds=bounds,
                args=(ring_sizes,),
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                mutation=(0.5, 1.5),
                recombination=0.8,
                init='sobol',
                polish=True,
                callback=lambda x, convergence: (time.time() > deadline)
            )
        except Exception:
            continue
        
        if result.success or result.fun < 1e5:
            centers = compute_centers(result.x, ring_sizes)
            if len(centers) == N:
                # Verify validity
                min_dist = np.inf
                for i in range(N):
                    for j in range(i + 1, N):
                        d = np.sqrt((centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2)
                        min_dist = min(min_dist, d)
                
                max_r = np.max(np.sqrt(centers[:, 0]**2 + centers[:, 1]**2))
                
                if max_r <= 1.0:
                    r = min_dist / 2.0
                    r = min(r, 1.0 - max_r)
                    if r > best_radius:
                        best_radius = r
                        best_points = centers
    
    # If no valid solution found, fall back to hexagonal lattice
    if best_points is None:
        # Simple hexagonal grid fallback
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                best_points = points[np.argsort(radius, kind="stable")[:N]]
                break
            spacing *= 0.98
            if spacing < 0.01:
                break
    
    return best_points
# EVOLVE-BLOCK-END
