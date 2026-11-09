"""
Strategy: Global Stochastic Search with Simulated Annealing.
The current program uses Lloyd's algorithm, which minimizes the sum of squared
distances (k-means objective) and gets trapped in local minima where the
covering radius (max distance) is not optimal. 

This program uses Simulated Annealing (SA) to directly minimize the covering
radius. 
1. We initialize with a Fibonacci spiral.
2. We iteratively pick a random point, perturb it by a random amount (temperature-dependent),
   and calculate the new covering radius.
3. We accept the move if it improves the radius, or with a probability based on
   the Boltzmann distribution if it worsens the radius (allowing escape from local minima).
4. The temperature cools over time, transitioning from exploration to exploitation.

This approach is structurally different from Lloyd's because:
- It directly optimizes the max-norm objective (covering radius) rather than mean-square.
- It is stochastic and can jump over energy barriers (local minima in the covering radius landscape).
- It does not rely on the convex hull structure for the update rule, only for evaluation.
"""

import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _covering_radius(points):
    """
    Compute the covering radius in degrees.
    Vectorized computation of circumradii of convex hull facets.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        p1 = points[facets[:, 0]]
        p2 = points[facets[:, 1]]
        p3 = points[facets[:, 2]]
        
        # The circumcenter of a spherical triangle is the unit vector c such that
        # c.p1 = c.p2 = c.p3.
        # This implies c is orthogonal to (p1-p2) and (p1-p3).
        # So c is parallel to cross(p1-p2, p1-p3).
        
        v1 = p1 - p2
        v2 = p1 - p3
        c_dir = np.cross(v1, v2)
        
        norms = np.linalg.norm(c_dir, axis=1)
        
        # Handle degenerate cases
        valid = norms > 1e-10
        if not np.any(valid):
            return 90.0 # Fallback
            
        c = np.zeros_like(c_dir)
        c[valid] = c_dir[valid] / norms[valid, np.newaxis]
        
        # Determine correct orientation: c should be "inside" the triangle.
        # Check dot product with centroid.
        centroids = (p1 + p2 + p3) / 3.0
        c_norms = np.linalg.norm(centroids, axis=1)
        valid_centroids = c_norms > 1e-10
        centroids[valid_centroids] /= c_norms[valid_centroids, np.newaxis]
        
        dots = np.sum(c * centroids, axis=1)
        c[dots < 0] *= -1
        
        # Compute angular distance from c to p1
        cos_theta = np.sum(c * p1, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        # Only consider valid facets
        theta[~valid] = 0.0
        
        return np.max(theta) * 180.0 / np.pi
        
    except Exception:
        return 180.0

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return points / norms

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    current_points = points.copy()
    current_radius = best_radius
    
    # Simulated Annealing parameters
    T_start = 1.0
    T_end = 1e-6
    # We want to run as many steps as possible.
    # Let's estimate steps. ConvexHull on 130 points is fast (~1-5ms).
    # If we can do 1000 steps in 10s, we can do ~10000 in 100s.
    # Let's set a high iteration count and let the deadline guard stop it.
    max_iterations = 20000
    
    rng = np.random.default_rng(42)
    
    for iter in range(max_iterations):
        if time.time() > deadline:
            break
            
        # Cooling schedule: Exponential
        # T(t) = T_start * (T_end/T_start)^(t/max_iterations)
        progress = iter / max_iterations
        T = T_start * (T_end / T_start) ** progress
        
        # Pick a random point to perturb
        idx = rng.integers(0, N)
        original_point = current_points[idx].copy()
        
        # Generate a random perturbation
        # Perturbation size should scale with T?
        # Let's use a fixed small step size for the direction, but maybe scale the probability?
        # Actually, standard SA: propose a move, accept based on delta E and T.
        # The "move" itself can be a random displacement.
        # Let's use a Gaussian perturbation in 3D space, then normalize.
        # Step size: maybe 0.01 to 0.1 radians?
        # Let's use a step size that depends on T to explore more when hot.
        step_size = 0.05 + 0.05 * T  # Between 0.05 and 0.1 radians roughly
        
        # Random direction
        random_dir = rng.normal(size=3)
        random_dir /= np.linalg.norm(random_dir)
        
        # Move point
        new_point = original_point + step_size * random_dir
        new_point /= np.linalg.norm(new_point)
        
        # Update current points
        current_points[idx] = new_point
        
        # Evaluate new radius
        new_radius = _covering_radius(current_points)
        
        # Acceptance criterion
        delta_E = new_radius - current_radius
        if delta_E < 0:
            # Improvement
            current_radius = new_radius
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = current_points.copy()
        else:
            # Worse: accept with probability exp(-delta_E / T)
            # delta_E is in degrees. We need to scale this for the Boltzmann factor.
            # A change of 0.1 degrees is significant.
            # Let's scale delta_E so that 0.1 deg corresponds to ~1 in the exponent at T=1.
            # So scale factor is 10.
            prob = np.exp(-10.0 * delta_E / T)
            if rng.random() < prob:
                current_radius = new_radius
            else:
                # Reject: revert
                current_points[idx] = original_point
    
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Covering radius: {_covering_radius(pts)} degrees")
    print(pts.shape)
