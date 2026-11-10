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
    
    def evaluate(params, n_per_ring):
        """
        params: [r_1, theta_1, r_2, theta_2, ...]
        n_per_ring: list of counts for rings 1..K
        Returns: negative of the max possible radius (to be minimized)
        """
        K = len(n_per_ring)
        points = np.zeros((N, 2))
        points[0] = [0.0, 0.0]
        
        idx = 1
        for i in range(K):
            n_i = n_per_ring[i]
            r_i = params[2*i]
            theta_i = params[2*i + 1]
            
            angles = theta_i + 2 * np.pi * np.arange(n_i) / n_i
            points[idx:idx+n_i, 0] = r_i * np.cos(angles)
            points[idx:idx+n_i, 1] = r_i * np.sin(angles)
            idx += n_i
        
        # Check if all points are in the unit circle
        dists = np.linalg.norm(points, axis=1)
        if np.any(dists > 1.0):
            return 1e6 # Penalty
            
        # Calculate min distance
        # Boundary constraint
        min_boundary = 1.0 - np.max(dists)
        
        # Pairwise distances
        tree = cKDTree(points)
        dists_nn, _ = tree.query(points, k=2)
        min_pair = np.min(dists_nn[:, 1])
        
        # The radius of the circles is half the minimum distance between centers
        # and also limited by the distance to the boundary.
        # r = min(min_pair / 2, min_boundary)
        # We want to maximize r.
        r = min(min_pair / 2.0, min_boundary)
        
        return -r

    best_r = 0.0
    best_points = None
    
    # Generate candidate ring structures
    # We want to distribute 299 circles into K rings.
    # Heuristic: n_i proportional to i (linear growth)
    
    for K in range(9, 13):
        # Base distribution
        total_weight = K * (K + 1) / 2.0
        n_per_ring = []
        for i in range(1, K + 1):
            count = int((N - 1) * i / total_weight)
            n_per_ring.append(max(1, count))
            
        # Adjust to sum to N-1
        current_sum = sum(n_per_ring)
        diff = (N - 1) - current_sum
        if diff != 0:
            for i in range(K - 1, -1, -1):
                if diff > 0:
                    n_per_ring[i] += 1
                    diff -= 1
                elif diff < 0:
                    if n_per_ring[i] > 1:
                        n_per_ring[i] -= 1
                        diff += 1
                if diff == 0:
                    break
        
        if sum(n_per_ring) != N - 1:
            continue
            
        # Initial guess for parameters
        init_params = np.zeros(2 * K)
        for i in range(K):
            init_params[2*i] = (i + 1) / (K + 1) * 0.95
            init_params[2*i + 1] = 0.0
            
        # Try a few different initial angles to escape local minima
        for angle_offset in [0.0, np.pi / 6.0]:
            p_init = init_params.copy()
            for i in range(K):
                p_init[2*i + 1] = angle_offset
                
            try:
                result = minimize(evaluate, p_init, args=(n_per_ring,),
                                  method='Nelder-Mead',
                                  options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10})
                
                r_val = -result.fun
                if r_val > best_r:
                    best_r = r_val
                    # Reconstruct points
                    points = np.zeros((N, 2))
                    points[0] = [0.0, 0.0]
                    idx = 1
                    for i in range(K):
                        n_i = n_per_ring[i]
                        r_i = result.x[2*i]
                        theta_i = result.x[2*i + 1]
                        angles = theta_i + 2 * np.pi * np.arange(n_i) / n_i
                        points[idx:idx+n_i, 0] = r_i * np.cos(angles)
                        points[idx:idx+n_i, 1] = r_i * np.sin(angles)
                        idx += n_i
                    best_points = points
                    
            except Exception:
                pass

    if best_points is None:
        # Fallback
        return np.zeros((N, 2))
        
    return best_points
# EVOLVE-BLOCK-END
