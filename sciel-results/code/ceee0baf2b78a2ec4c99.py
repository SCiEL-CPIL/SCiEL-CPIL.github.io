"""Rewritten program: Simulated Annealing minimization of logarithmic energy on the sphere."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math

def _fibonacci_sphere(n):
    """Generate a Fibonacci spiral distribution on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _pairwise_distances(pts):
    """Compute pairwise distances matrix."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return dist

def _energy_from_dist(dist):
    """Compute total energy from distance matrix."""
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    d = np.clip(d, 1e-12, None)
    return -np.sum(np.log(d))

def construct():
    # Initialize with Fibonacci sphere
    pts = _fibonacci_sphere(N).astype(np.float64)
    
    # Precompute initial distances and energy
    dist_mat = _pairwise_distances(pts)
    current_energy = _energy_from_dist(dist_mat)
    
    # Simulated Annealing parameters
    T_start = 0.5
    T_end = 1e-8
    n_iterations = 500000
    
    # Geometric cooling
    cool_factor = (T_end / T_start) ** (1.0 / n_iterations)
    T = T_start
    
    rng = np.random.default_rng(42)
    
    best_pts = pts.copy()
    best_energy = current_energy
    
    # Perturbation scale
    sigma = 0.5
    
    for step in range(n_iterations):
        # Pick a random point to move
        i = rng.integers(0, N)
        
        # Store original position and distances to others
        orig_pt = pts[i]
        orig_dists = dist_mat[i]
        
        # Generate a random perturbation on the sphere
        # Small random vector
        delta = rng.normal(0, sigma, 3)
        new_pt = orig_pt + delta
        # Normalize to unit sphere
        norm = np.linalg.norm(new_pt)
        if norm < 1e-12:
            new_pt = orig_pt
        else:
            new_pt = new_pt / norm
            
        # If the point didn't change significantly, skip
        if np.allclose(orig_pt, new_pt, atol=1e-10):
            T *= cool_factor
            continue
            
        # Calculate new distances from i to all j
        diff = new_pt - pts
        new_dists = np.linalg.norm(diff, axis=1)
        new_dists[i] = 0.0 # self distance
        
        # Calculate Delta Energy
        # Only terms involving i change.
        mask = np.ones(N, dtype=bool)
        mask[i] = False
        
        old_d = orig_dists[mask]
        new_d = new_dists[mask]
        
        old_d = np.clip(old_d, 1e-12, None)
        new_d = np.clip(new_d, 1e-12, None)
        
        delta_E = np.sum(np.log(old_d)) - np.sum(np.log(new_d))
        
        # Acceptance criterion
        if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
            # Accept move
            pts[i] = new_pt
            # Update dist_mat
            dist_mat[i, :] = new_dists
            dist_mat[:, i] = new_dists
            
            current_energy += delta_E
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
                
        T *= cool_factor
        
    # Final cleanup
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
