# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses Simulated Annealing on spherical coordinates (points constrained to unit sphere).
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = 500.0  # Safety margin before 600s limit
    
    # State: spherical coordinates (theta, phi) for each point
    # theta: [0, 2*pi], phi: [0, pi]
    # We will optimize these 2*n parameters.
    
    def to_cartesian(sph_coords):
        # sph_coords: (n, 2) array of [theta, phi]
        theta = sph_coords[:, 0]
        phi = sph_coords[:, 1]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.column_stack((x, y, z))
    
    def metric(points):
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    # Initialize with random points on the sphere
    np.random.seed(42)
    current_sph = np.random.rand(n, 2) * np.array([2 * np.pi, np.pi])
    current_pts = to_cartesian(current_sph)
    current_score = metric(current_pts)
    
    best_sph = current_sph.copy()
    best_pts = current_pts.copy()
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 1.0  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.999
    max_iter = 500000
    
    i = 0
    while i < max_iter and (time.time() - start_time) < deadline:
        # Perturb one random point
        idx = np.random.randint(n)
        delta_theta = np.random.normal(0, 0.1)
        delta_phi = np.random.normal(0, 0.1)
        
        new_sph = current_sph.copy()
        new_sph[idx, 0] = (current_sph[idx, 0] + delta_theta) % (2 * np.pi)
        new_sph[idx, 1] = np.clip(current_sph[idx, 1] + delta_phi, 0, np.pi)
        
        new_pts = to_cartesian(new_sph)
        new_score = metric(new_pts)
        
        # Metropolis criterion
        if new_score > current_score or np.random.rand() < np.exp((new_score - current_score) / T):
            current_sph = new_sph
            current_pts = new_pts
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_sph = current_sph.copy()
                best_pts = current_pts.copy()
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
        
        i += 1
        
        # Optional: Restart if stuck? For now, just let it run.
        # If we want to be more robust, we could add multi-start.
        # But SA with long run is usually good.
        
    # Ensure the result is valid
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (14, 3):
        # Fallback: Random points on sphere
        fallback_sph = np.random.rand(n, 2) * np.array([2 * np.pi, np.pi])
        return to_cartesian(fallback_sph)
        
    return best_pts

# EVOLVE-BLOCK-END
