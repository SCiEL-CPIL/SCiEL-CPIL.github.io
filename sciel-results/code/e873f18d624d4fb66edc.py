# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform
import time

N = 50

def _compute_radius(positions):
    """Compute the maximum radius for given positions."""
    n = len(positions)
    if n == 0:
        return 0.0
    
    # Distance to boundary
    norms = np.linalg.norm(positions, axis=1)
    boundary_dist = 1.0 - norms
    
    # Minimum pairwise distance
    if n > 1:
        dists = pdist(positions, metric='euclidean')
        min_pair_dist = np.min(dists)
    else:
        min_pair_dist = np.inf
        
    return min(np.min(boundary_dist), min_pair_dist / 2.0)

def _initial_positions(n, seed=42):
    """Generate initial positions using a hexagonal lattice."""
    rng = np.random.default_rng(seed)
    
    # Hexagonal packing is the densest in 2D.
    # We generate a large hex grid and pick the n points closest to the center.
    points = []
    # Radius of the container is 1.
    # Hex spacing s. Distance between centers is s.
    # We want to fit ~50 points. Area of unit circle is pi.
    # Area per hex point is s^2 * sqrt(3)/2.
    # 50 * s^2 * sqrt(3)/2 = pi => s^2 = 2*pi / (50*sqrt(3)) ~ 0.073 => s ~ 0.27.
    # But we need to fit them inside, so effective radius is 1 - r.
    # Let's just generate a grid with spacing 0.2 and filter.
    
    spacing = 0.18
    for row in range(-10, 11):
        for col in range(-10, 11):
            x = col * spacing * 1.5
            y = row * spacing * np.sqrt(3) + (col % 2) * spacing * np.sqrt(3) / 2
            if x*x + y*y < 0.9:
                points.append([x, y])
                
    points = np.array(points)
    
    if len(points) < n:
        # Fallback to random if grid is too sparse
        positions = rng.uniform(-0.8, 0.8, size=(n, 2))
        for i in range(n):
            while np.linalg.norm(positions[i]) > 0.8:
                positions[i] = rng.uniform(-0.8, 0.8, size=2)
        return positions

    # Select n points closest to center
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)[:n]
    positions = points[indices].copy()
    
    # Add small jitter
    positions += rng.normal(0, 0.005, size=positions.shape)
    
    return positions

def _optimize(positions, max_time=100.0, seed=42):
    """
    Optimizes positions using vectorized force-based local search.
    This is much faster than Python loops for N=50.
    """
    rng = np.random.default_rng(seed)
    positions = positions.copy()
    n = len(positions)
    
    best_radius = _compute_radius(positions)
    best_positions = positions.copy()
    
    start_time = time.time()
    lr = 0.05
    decay = 0.9995
    
    while time.time() - start_time < max_time:
        # Compute all pairwise distances
        # diff[i, j] = positions[i] - positions[j]
        # Using broadcasting is memory heavy for large N, but N=50 is fine (50x50x2)
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        
        # Avoid division by zero
        dist_sq[dist_sq < 1e-12] = 1e-12
        
        # Force magnitude: 1/dist^2
        # We only want repulsion for close pairs to avoid long-range instability?
        # Actually, for packing, we want to maximize the minimum distance.
        # A common potential is 1/r. Force is -d/dr (1/r) = 1/r^2.
        # We want to push particles apart.
        # Force on i due to j is proportional to (pos_i - pos_j) / dist_ij^2
        
        # We sum forces from all other particles.
        # F_i = sum_j (pos_i - pos_j) / dist_ij^2
        # Note: dist_sq[i,i] is 0 (set to 1e-12), so term is 0.
        
        inv_dist_sq = 1.0 / dist_sq
        # Zero out diagonal
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        forces = np.sum(diff * inv_dist_sq[..., np.newaxis], axis=1)
        
        # Wall repulsion
        # If a particle is close to the boundary, push it inward.
        norms = np.linalg.norm(positions, axis=1)
        # We want to keep them at least 'r' away from boundary.
        # We don't know r exactly, but we can use a heuristic.
        # If norm > 1 - best_radius, push in.
        # Actually, just push in if norm > 0.95 or so.
        # Let's use a soft wall: force = -k * (norm - (1-r_est)) * pos/norm
        # Simpler: just penalize norm > 1 - current_min_radius/2?
        # Let's use a fixed margin for stability.
        margin = 0.13 # Approx radius
        wall_force = np.zeros_like(positions)
        for i in range(n):
            if norms[i] > 1.0 - margin:
                # Push towards center
                strength = (norms[i] - (1.0 - margin)) * 10.0
                wall_force[i] = -positions[i] / norms[i] * strength
        forces += wall_force
        
        # Update positions
        positions += lr * forces
        
        # Hard constraint: keep inside
        new_norms = np.linalg.norm(positions, axis=1)
        mask = new_norms > 1.0
        if np.any(mask):
            positions[mask] = positions[mask] / new_norms[mask, np.newaxis] * 0.999
            
        current_radius = _compute_radius(positions)
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_positions = positions.copy()
            
        lr *= decay
        if lr < 1e-5:
            break
            
    return best_positions, best_radius

def construct():
    """Construct centers of 50 equal circles inside the unit circle."""
    best_positions = None
    best_radius = 0.0
    
    # Multi-start strategy
    # Run several independent optimization sequences
    # Time budget: 120s total. Leave 20s for safety.
    # 5 starts, 20s each = 100s.
    
    seeds = [42, 123, 456, 789, 101]
    
    for seed in seeds:
        positions = _initial_positions(N, seed=seed)
        positions, radius = _optimize(positions, max_time=20.0, seed=seed)
        
        if radius > best_radius:
            best_radius = radius
            best_positions = positions.copy()
            
    # Final polish on the best result
    if best_positions is not None:
        best_positions, best_radius = _optimize(best_positions, max_time=10.0, seed=999)
        
    # Ensure valid output
    if best_positions is None:
        best_positions = _initial_positions(N, seed=0)
        
    # Final boundary check
    norms = np.linalg.norm(best_positions, axis=1)
    mask = norms > 0.999
    if np.any(mask):
        best_positions[mask] = best_positions[mask] / norms[mask, np.newaxis] * 0.999
        
    return best_positions
# EVOLVE-BLOCK-END
