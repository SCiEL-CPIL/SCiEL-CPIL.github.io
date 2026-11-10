"""Ring-based packing for 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def construct():
    """
    Place 300 equal circles inside a unit circle using a ring-based representation.
    Circles are placed on concentric rings with optimized radii and angular offsets.
    """
    
    def compute_min_distance(params, n_per_ring, n_rings):
        """
        params: array of length 2*(n_rings-1) for rings 1..n_rings-1
                [r_1, phi_1, r_2, phi_2, ...]
        n_per_ring: list/array of n_i for ring i (i=0 is center, n_0=1)
        n_rings: total number of rings
        Returns: minimum radius (half of min distance between centers, or distance to boundary)
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
        min_dist = 1.0  # initial guess
        
        # Distance to boundary
        min_dist = min(min_dist, 1.0 - np.max(dists))
        
        # Pairwise distances using cKDTree
        tree = cKDTree(points)
        distances, indices = tree.query(points, k=2)
        min_nn = np.min(distances[:, 1])
        min_dist = min(min_dist, min_nn / 2.0)
        
        return min_dist
    
    def negative_min_distance(params, n_per_ring, n_rings):
        """Negative of min distance for minimization."""
        return -compute_min_distance(params, n_per_ring, n_rings)
    
    def get_points_from_params(params, n_per_ring, n_rings):
        """Reconstruct points from parameters."""
        points = np.zeros((N, 2))
        points[0] = [0.0, 0.0]
        idx = 1
        for i in range(1, n_rings):
            n_i = n_per_ring[i]
            r_i = params[2*(i-1)]
            phi_i = params[2*(i-1) + 1]
            angles = phi_i + 2 * np.pi * np.arange(n_i) / n_i
            points[idx:idx+n_i, 0] = r_i * np.cos(angles)
            points[idx:idx+n_i, 1] = r_i * np.sin(angles)
            idx += n_i
        return points
    
    best_r = 0.0
    best_points = None
    
    # Try different numbers of rings
    for n_rings in range(6, 15):
        # Distribute N-1 circles among n_rings-1 rings
        # Use linear distribution: n_i proportional to i
        c = 2.0 * (N - 1) / ((n_rings - 1) * n_rings)
        n_per_ring = [1]  # center
        for i in range(1, n_rings):
            n_i = max(1, int(round(c * i)))
            n_per_ring.append(n_i)
        
        # Adjust to make sum exactly N
        total = sum(n_per_ring)
        if total != N:
            diff = N - total
            # Adjust the largest ring first, then others
            for j in range(n_rings - 1, 0, -1):
                if diff > 0:
                    n_per_ring[j] += 1
                    diff -= 1
                elif diff < 0:
                    if n_per_ring[j] > 1:
                        n_per_ring[j] -= 1
                        diff += 1
        
        # Multiple random restarts for better exploration
        for restart in range(5):
            # Initial guess for parameters
            init_params = np.zeros(2 * (n_rings - 1))
            for i in range(1, n_rings):
                # Radii roughly proportional to i / (n_rings - 1)
                base_r = i / (n_rings - 1) * 0.95
                if restart == 0:
                    init_params[2*(i-1)] = base_r
                else:
                    # Perturb slightly
                    init_params[2*(i-1)] = base_r * (1 + 0.1 * np.random.randn())
                # Angular offsets
                if restart == 0:
                    init_params[2*(i-1) + 1] = 0.0
                else:
                    init_params[2*(i-1) + 1] = np.random.uniform(0, 2*np.pi)
            
            # Optimize
            try:
                result = minimize(negative_min_distance, init_params, args=(n_per_ring, n_rings),
                                 method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10})
                
                r_val = -result.fun
                if r_val > best_r and r_val > 0:
                    best_r = r_val
                    best_points = get_points_from_params(result.x, n_per_ring, n_rings)
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
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            if len(points) >= N:
                # Sort by radius and take first N
                order = np.argsort(radius[mask], kind="stable")[:N]
                return points[order]
            spacing *= 0.98
    
    return best_points
# EVOLVE-BLOCK-END
