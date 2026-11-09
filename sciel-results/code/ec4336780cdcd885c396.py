"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START

def _fibonacci_sphere(n):
    """Generate points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points

def _covering_radius(points):
    """
    Estimate the covering radius.
    The covering radius is the maximum distance from any point on the sphere to the nearest point in the set.
    This is equivalent to the maximum circumradius of the facets of the convex hull (spherical Voronoi cells).
    A good approximation is the maximum over all points i of the distance to its nearest neighbor? No, that's packing radius.
    Covering radius is larger.
    For a set of points, the covering radius is the radius of the largest empty cap.
    This is determined by the centers of the Voronoi cells.
    Since we don't have an easy spherical Voronoi implementation, we can approximate by checking the distance to the nearest neighbor for each point, 
    but that underestimates the covering radius.
    
    Actually, for the purpose of the score, the evaluator will calculate the exact covering radius.
    We just need to minimize it.
    Minimizing the maximum distance to the nearest neighbor (packing radius) is a good proxy, but not identical.
    However, for well-distributed points, they are correlated.
    
    Let's use a better proxy: 
    The covering radius is the maximum, over all points on the sphere, of the distance to the nearest generator.
    This maximum is attained at a vertex of the Voronoi diagram, i.e., a point equidistant to 3 (or more) generators.
    Without Voronoi, we can't easily find these.
    
    However, the repulsive force method tends to equalize the distances, leading to a near-regular triangulation, 
    where the covering radius is close to the circumradius of the triangles.
    
    We will use the maximum-minimum distance (packing radius) as the optimization objective, 
    as it is smooth (with the log-sum-exp trick) and leads to good coverings.
    """
    pass

def construct():
    # Start with a good initial guess
    points = _fibonacci_sphere(N)
    
    # Optimization parameters
    p = 20  # Power for the sum approximation of min distance
    eta = 0.01  # Initial step size
    min_eta = 1e-6
    decay = 0.999
    
    start_time = time.time()
    time_limit = 120.0  # Leave some margin for 150s limit
    
    best_points = points.copy()
    best_score = 0.0  # We want to maximize the minimum distance
    
    # Precompute indices for i != j
    mask = ~np.eye(N, dtype=bool)
    i_idx, j_idx = np.where(mask)
    
    for step in range(100000):
        if time.time() - start_time > time_limit:
            break
            
        # Compute pairwise distances
        # points shape (N, 3)
        # diff shape (N, N, 3)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero (though points shouldn't be identical)
        dists[i_idx, j_idx] = np.maximum(dists[i_idx, j_idx], 1e-10)
        
        # Compute forces: gradient of -sum_{i!=j} d_ij^{-p}
        # We want to maximize min d_ij, so we minimize sum d_ij^{-p}
        # grad_{P_i} (d_ij^{-p}) = -p * d_ij^{-p-2} * (P_i - P_j)
        # So descent step is P_i += eta * p * d_ij^{-p-2} * (P_i - P_j)
        
        # Vectorized calculation
        # weights shape (N, N)
        weights = p * (dists ** (-p - 2))
        np.fill_diagonal(weights, 0.0)
        
        # Force on i is sum_j weights[i,j] * (P_i - P_j)
        # = sum_j weights[i,j] * P_i - sum_j weights[i,j] * P_j
        # = P_i * sum_j weights[i,j] - sum_j (weights[i,j] * P_j)
        
        sum_weights = np.sum(weights, axis=1)
        # weighted_points shape (N, 3)
        weighted_points = np.sum(weights[:, :, np.newaxis] * points[np.newaxis, :, :], axis=1)
        
        forces = points * sum_weights[:, np.newaxis] - weighted_points
        
        # Update points
        points = points + eta * forces
        
        # Normalize
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        # Check score (packing radius proxy)
        # Recalculate min distance
        min_dist = np.min(dists[i_idx, j_idx])
        
        if min_dist > best_score:
            best_score = min_dist
            best_points = points.copy()
            
        # Decay step size
        eta *= decay
        if eta < min_eta:
            # Try a few more steps with min_eta or break
            if step > 10000:
                break

    return best_points

# EVOLVE-BLOCK-END
