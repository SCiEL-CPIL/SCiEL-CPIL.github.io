"""Ring-based packing for 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Place 300 equal circles inside a unit circle using a ring-based representation.
    Circles are placed on concentric rings with optimized radii and angular offsets.
    """
    
    # We'll use a ring-based approach.
    # Let's say we have R rings. Ring 0 is the center (1 circle), rings 1..R-1 have more circles.
    # For ring i (i >= 1), we place n_i circles at radius r_i with angular offset phi_i.
    
    # Strategy:
    # 1. Choose a number of rings and distribution of circles per ring.
    # 2. Optimize the ring radii and angular offsets to maximize the minimum distance.
    
    # Let's try a heuristic for the number of rings and circles per ring.
    # Total circles = 1 (center) + sum(n_i for i in 1..R-1)
    
    # A common pattern: rings with increasing number of circles.
    # Let's try to find a good configuration by searching over the number of rings.
    
    def compute_min_distance(params, n_per_ring, n_rings):
        """
        params: array of length 2*(n_rings-1) for rings 1..n_rings-1
                [r_1, phi_1, r_2, phi_2, ...]
        n_per_ring: list/array of n_i for ring i (i=0 is center, n_0=1)
        n_rings: total number of rings
        Returns: minimum distance (half of min distance between centers, or distance to boundary)
        """
        # Build all points
        points = np.zeros((N, 2))
        points[0] = [0.0, 0.0]  # center
        
        idx = 1
        for i in range(1, n_rings):
            n_i = n_per_ring[i]
            r_i = params[2*(i-1)]
            phi_i = params[2*(i-1) + 1]
            angles = phi_i + 2 * np.pi * np.arange(n_i) / n_i
            points[idx:idx+n_i, 0] = r_i * np.cos(angles)
            points[idx:idx+n_i, 1] = r_i * np.sin(angles)
            idx += n_i
        
        # Check all points are within unit circle
        dists = np.linalg.norm(points, axis=1)
        if np.any(dists >= 1.0):
            return -1.0
        
        # Compute min distance between all pairs
        # Use a more efficient method for large N
        min_dist = 1.0  # initial guess
        
        # Distance to boundary
        min_dist = min(min_dist, 1.0 - np.max(dists))
        
        # Pairwise distances
        # For efficiency, we can use scipy's cKDTree or just compute all pairs for N=300
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        # Query all pairs
        distances, indices = tree.query(points, k=2)
        # distances[:, 1] is the nearest neighbor distance
        min_nn = np.min(distances[:, 1])
        min_dist = min(min_dist, min_nn / 2.0)
        
        return min_dist
    
    def negative_min_distance(params, n_per_ring, n_rings):
        """Negative of min distance for minimization."""
        return -compute_min_distance(params, n_per_ring, n_rings)
    
    # Try different numbers of rings and distributions
    best_r = 0.0
    best_points = None
    
    # Heuristic: try rings with approximately proportional numbers of circles
    # Ring i has roughly proportional to (2i+1) or similar
    
    # Let's try a few configurations
    for n_rings in range(5, 15):
        # Distribute N-1 circles among n_rings-1 rings
        # Use a heuristic: n_i proportional to i (or 2i-1)
        
        # Try linear distribution: n_i = c * i
        # Sum_{i=1}^{n_rings-1} c*i = N-1
        # c * (n_rings-1)*n_rings/2 = N-1
        # c = 2*(N-1) / ((n_rings-1)*n_rings)
        
        c = 2.0 * (N - 1) / ((n_rings - 1) * n_rings)
        n_per_ring = [1]  # center
        for i in range(1, n_rings):
            n_i = max(1, int(round(c * i)))
            n_per_ring.append(n_i)
        
        # Adjust to make sum exactly N
        total = sum(n_per_ring)
        if total != N:
            # Adjust the largest ring
            diff = N - total
            n_per_ring[-1] += diff
            if n_per_ring[-1] < 1:
                # Try adjusting a different ring
                for j in range(n_rings - 2, 0, -1):
                    n_per_ring[j] += 1
                    diff -= 1
                    if diff == 0:
                        break
        
        # Initial guess for parameters
        # Radii roughly proportional to i / (n_rings - 1)
        init_params = np.zeros(2 * (n_rings - 1))
        for i in range(1, n_rings):
            init_params[2*(i-1)] = i / (n_rings - 1) * 0.95  # radius
            init_params[2*(i-1) + 1] = 0.0  # angle
        
        # Optimize
        try:
            result = minimize(negative_min_distance, init_params, args=(n_per_ring, n_rings),
                             method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            r_val = -result.fun
            if r_val > best_r:
                best_r = r_val
                # Reconstruct points
                points = np.zeros((N, 2))
                points[0] = [0.0, 0.0]
                idx = 1
                for i in range(1, n_rings):
                    n_i = n_per_ring[i]
                    r_i = result.x[2*(i-1)]
                    phi_i = result.x[2*(i-1) + 1]
                    angles = phi_i + 2 * np.pi * np.arange(n_i) / n_i
                    points[idx:idx+n_i, 0] = r_i * np.cos(angles)
                    points[idx:idx+n_i, 1] = r_i * np.sin(angles)
                    idx += n_i
                best_points = points
        except Exception:
            pass
    
    # If optimization didn't find anything good, fall back to a simple hexagonal lattice
    if best_points is None or best_r < 0.04:
        # Hexagonal lattice
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
    
    return best_points
# EVOLVE-BLOCK-END
