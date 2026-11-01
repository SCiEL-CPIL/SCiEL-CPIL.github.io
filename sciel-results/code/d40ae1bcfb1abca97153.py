import numpy as np
import time
from scipy.spatial.distance import cdist

N = 100

def _compute_radius(centres):
    """Compute the maximum radius for given centres."""
    # Distance to walls
    wall_dists = np.min(np.abs(centres), axis=1)
    wall_min = np.min(wall_dists)
    
    # Pairwise distances
    dist_matrix = cdist(centres, centres)
    np.fill_diagonal(dist_matrix, np.inf)
    pair_min = np.min(dist_matrix) / 2.0
    
    return min(wall_min, pair_min)

def _relax(centres, n_iters=3000, dt=0.001, deadline=None):
    """Force-directed relaxation to maximize minimum distance."""
    centres = centres.copy()
    best_radius = _compute_radius(centres)
    best_centres = centres.copy()
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute pairwise distances
        dist_matrix = cdist(centres, centres)
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Vectorized forces
        forces = np.zeros_like(centres)
        
        # Pairwise repulsion: F_i = sum_{j != i} (c_i - c_j) / |c_i - c_j|^2 * mask
        # This is equivalent to -sum_{j != i} (c_j - c_i) / |c_i - c_j|^2
        # Let's compute: for each i, sum over j of (c_i - c_j) / d_ij^2
        # = c_i * sum(1/d^2) - sum(c_j / d^2)
        
        # Use safe division
        safe_dist = np.where(dist_matrix < 1e-10, 1e-10, dist_matrix)
        inv_dist_sq = 1.0 / (safe_dist ** 2)
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        # Only consider pairs within interaction range
        mask = (dist_matrix < 0.35) & (dist_matrix > 1e-10)
        inv_dist_sq_masked = np.where(mask, inv_dist_sq, 0.0)
        
        # F_i = sum_j (c_i - c_j) * inv_dist_sq[i,j]
        # = c_i * sum_j(inv_dist_sq[i,j]) - sum_j(c_j * inv_dist_sq[i,j])
        sum_inv = np.sum(inv_dist_sq_masked, axis=1)
        weighted_centres = inv_dist_sq_masked @ centres  # shape (N, 3)
        
        forces += centres * sum_inv[:, None] - weighted_centres
        
        # Wall repulsion
        for k in range(3):
            dist_to_neg = centres[:, k] + 0.5
            dist_to_pos = 0.5 - centres[:, k]
            
            # Only apply if close to wall
            mask_neg = dist_to_neg < 0.15
            mask_pos = dist_to_pos < 0.15
            
            forces[mask_neg, k] += 1.0 / (dist_to_neg[mask_neg] + 1e-10)
            forces[mask_pos, k] -= 1.0 / (dist_to_pos[mask_pos] + 1e-10)
        
        # Weak center attraction
        forces -= 0.003 * centres
        
        # Update positions
        centres += dt * forces
        
        # Clip to valid range
        centres = np.clip(centres, -0.49, 0.49)
        
        # Check radius
        radius = _compute_radius(centres)
        if radius > best_radius:
            best_radius = radius
            best_centres = centres.copy()
    
    return best_centres, best_radius

def _make_initial(nx, ny, nz):
    """Create initial grid configuration."""
    x = np.linspace(-0.4, 0.4, nx)
    y = np.linspace(-0.4, 0.4, ny)
    z = np.linspace(-0.4, 0.4, nz)
    
    points = []
    for zi in z:
        for yi in y:
            for xi in x:
                points.append([xi, yi, zi])
    
    points = np.array(points)
    return points[:N]

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 135s deadline, 15s safety margin
    
    # Initial configuration
    base_centres = _make_initial(5, 5, 4)
    
    best_centres = base_centres.copy()
    best_radius = _compute_radius(base_centres)
    
    # Try multiple restarts with different perturbations
    np.random.seed(12345)
    
    for restart in range(15):
        if time.time() > deadline - 10:
            break
        
        # Add random perturbation
        perturbation = np.random.randn(N, 3) * 0.03
        centres = base_centres + perturbation
        centres = np.clip(centres, -0.45, 0.45)
        
        # Relax
        relaxed, radius = _relax(centres, n_iters=2000, dt=0.002, deadline=deadline - 10)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = relaxed.copy()
    
    # Final polish if time allows
    if time.time() < deadline - 5 and best_radius > 0.105:
        polished, polished_radius = _relax(best_centres, n_iters=3000, dt=0.0005, deadline=deadline - 2)
        if polished_radius > best_radius:
            best_radius = polished_radius
            best_centres = polished.copy()
    
    return best_centres
