"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random

def construct():
    """
    Construct centers of 100 equal spheres in a unit cube [-0.5, 0.5]^3
    to maximize the minimum distance between any two centers and the
    distance to the cube boundary.
    
    Uses a multi-start local optimization approach:
    1. Initialize with a jittered grid
    2. Use scipy.optimize.minimize to maximize the minimum distance
    3. Try multiple random restarts and keep the best result
    """
    random.seed(42)
    np.random.seed(42)
    
    best_centers = None
    best_score = 0.0
    
    def score_function(centers):
        """Compute the packing score (minimum radius)."""
        if centers.shape != (N, 3):
            return 0.0
        # Check if all points are inside the cube
        if np.any(np.abs(centers) > 0.5 + 1e-9):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        
        # Compute all pairwise distances
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair_dist = np.min(dist_matrix)
        
        # Distance to boundary
        boundary_dists = 0.5 - np.abs(centers)
        min_boundary = np.min(boundary_dists)
        
        return min(min_pair_dist / 2.0, min_boundary)
    
    def negative_score(centers_flat):
        """Objective function to minimize (negative of score)."""
        centers = centers_flat.reshape(N, 3)
        # Add penalty for going out of bounds
        out_of_bounds = np.maximum(0, np.abs(centers) - 0.5)
        penalty = 100.0 * np.sum(out_of_bounds ** 2)
        
        if centers.shape != (N, 3):
            return 1e10
        if not np.all(np.isfinite(centers)):
            return 1e10
        
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair_dist = np.min(dist_matrix)
        
        boundary_dists = 0.5 - np.abs(centers)
        min_boundary = np.min(boundary_dists)
        
        raw_score = min(min_pair_dist / 2.0, min_boundary)
        return -raw_score + penalty
    
    def run_optimization(initial_centers, max_iter=200):
        """Run optimization from a given initial configuration."""
        x0 = initial_centers.flatten()
        
        # Use Nelder-Mead for derivative-free optimization
        result = minimize(
            negative_score,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': max_iter,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        
        centers = result.x.reshape(N, 3)
        
        # Project back into the cube
        centers = np.clip(centers, -0.5, 0.5)
        
        return centers, score_function(centers)
    
    # Strategy 1: Jittered grid initialization
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial_grid = grid[:N].copy()
    
    # Add small random jitter
    jittered = initial_grid + np.random.uniform(-0.02, 0.02, size=(N, 3))
    jittered = np.clip(jittered, -0.5, 0.5)
    
    centers, sc = run_optimization(jittered, max_iter=300)
    if sc > best_score:
        best_score = sc
        best_centers = centers.copy()
    
    # Strategy 2: Random initializations with local optimization
    for restart in range(5):
        # Random initialization
        initial = np.random.uniform(-0.45, 0.45, size=(N, 3))
        
        # Run a few iterations of a simple repulsion-based relaxation
        for it in range(50):
            # Compute forces
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(initial, initial)
            np.fill_diagonal(dist_matrix, np.inf)
            
            # Repulsive force: push apart close points
            force = np.zeros_like(initial)
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    d = dist_matrix[i, j]
                    if d < 0.2:
                        # Strong repulsion
                        diff = initial[i] - initial[j]
                        force[i] += diff / (d ** 3) * 0.01
            
            # Boundary repulsion
            for i in range(N):
                for k in range(3):
                    if initial[i, k] > 0.4:
                        force[i, k] -= (initial[i, k] - 0.4) * 0.5
                    elif initial[i, k] < -0.4:
                        force[i, k] -= (initial[i, k] + 0.4) * 0.5
            
            initial += force * 0.01
            initial = np.clip(initial, -0.5, 0.5)
        
        # Now run local optimization
        centers, sc = run_optimization(initial, max_iter=200)
        if sc > best_score:
            best_score = sc
            best_centers = centers.copy()
    
    # Strategy 3: Lattice-based initialization with different symmetries
    # FCC-like structure
    a = 1.0 / (N ** (1.0/3.0)) * 0.9
    fcc_points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                fcc_points.append([
                    (i * a) - 0.5 + a/2,
                    (j * a) - 0.5 + a/2,
                    (k * a) - 0.5 + a/2
                ])
                # Add face-centered points
                fcc_points.append([
                    (i * a) - 0.5 + a,
                    (j * a) - 0.5 + a/2,
                    (k * a) - 0.5 + a/2
                ])
                fcc_points.append([
                    (i * a) - 0.5 + a/2,
                    (j * a) - 0.5 + a,
                    (k * a) - 0.5 + a/2
                ])
                fcc_points.append([
                    (i * a) - 0.5 + a/2,
                    (j * a) - 0.5 + a/2,
                    (k * a) - 0.5 + a
                ])
    
    fcc_points = np.array(fcc_points)
    # Filter to those inside the cube
    mask = np.all(np.abs(fcc_points) <= 0.5, axis=1)
    fcc_points = fcc_points[mask]
    
    if len(fcc_points) >= N:
        # Select N points
        idx = np.random.choice(len(fcc_points), N, replace=False)
        fcc_initial = fcc_points[idx] + np.random.uniform(-0.01, 0.01, size=(N, 3))
        fcc_initial = np.clip(fcc_initial, -0.5, 0.5)
        
        centers, sc = run_optimization(fcc_initial, max_iter=200)
        if sc > best_score:
            best_score = sc
            best_centers = centers.copy()
    
    # Final polish: one more round of optimization on the best solution
    if best_centers is not None:
        final_centers, final_sc = run_optimization(best_centers, max_iter=300)
        if final_sc > best_score:
            best_score = final_sc
            best_centers = final_centers
    
    # Ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

# EVOLVE-BLOCK-END
