"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial.distance import cdist

N = 200

# EVOLVE-BLOCK-START
def construct():
    """
    Pack 200 equal circles in the unit square [-0.5, 0.5]^2 to maximize the common radius.
    
    Strategy:
    1. Start with a hexagonal packing (better than square grid)
    2. Optimize using a repulsive force model to maximize minimum distance
    3. Use multiple restarts with different initial configurations
    4. Deadline guard to return best result before 150s
    """
    start_time = time.time()
    deadline = start_time + 130  # Safety margin: 20s before 150s limit
    
    def compute_radius(positions):
        """Compute the maximum possible radius given positions."""
        # Check boundary constraints
        min_coord = -0.5 - positions.min(axis=0)
        max_coord = 0.5 - positions.max(axis=0)
        boundary_margin = min(min_coord.min(), max_coord.min())
        
        if boundary_margin < 0:
            return 0.0
        
        # Check pairwise distances
        dist_matrix = cdist(positions, positions)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair_dist = dist_matrix.min()
        
        # The radius is limited by both boundary and pairwise constraints
        return min(boundary_margin, min_pair_dist / 2.0)
    
    def hexagonal_initialization(n_circles):
        """Create initial positions using hexagonal packing."""
        positions = []
        # Hexagonal packing has rows offset by half spacing
        # Try to fit as many as possible in a hexagonal grid
        spacing = 0.048
        for row in range(50):
            for col in range(50):
                if len(positions) >= n_circles:
                    break
                x = (col + (row % 2) * 0.5) * spacing
                y = row * spacing * np.sqrt(3) / 2
                positions.append((x, y))
            if len(positions) >= n_circles:
                break
        
        positions = np.array(positions[:n_circles])
        # Center the grid
        positions -= positions.mean(axis=0)
        # Scale to fit within bounds
        max_abs = np.abs(positions).max()
        if max_abs > 0:
            scale = min(0.49 / max_abs, 1.0)
            positions *= scale
        return positions
    
    def square_grid_initialization(n_circles):
        """Create initial positions using square grid."""
        m = int(np.ceil(np.sqrt(n_circles)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n_circles]
    
    def random_initialization(n_circles, seed=None):
        """Create random initial positions."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        
        positions = rng.uniform(-0.45, 0.45, size=(n_circles, 2))
        return positions
    
    def optimize_repulsion(positions, max_steps=5000, step_size=0.001):
        """
        Optimize positions using repulsive forces.
        Particles repel each other and are attracted to the center of the square.
        """
        positions = positions.copy()
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Compute all pairwise distances
            dist_matrix = cdist(positions, positions)
            np.fill_diagonal(dist_matrix, np.inf)
            
            # Compute repulsive forces (inverse square law, capped at minimum distance)
            min_dist = 0.01
            dist_capped = np.maximum(dist_matrix, min_dist)
            force_matrix = 1.0 / (dist_capped ** 2)
            np.fill_diagonal(force_matrix, 0.0)
            
            # Sum forces for each particle
            # Force direction is from j to i: (pos_i - pos_j) / dist
            # F_i = sum_j (pos_i - pos_j) / dist_ij^3
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            # dist_matrix has shape (N, N)
            # We want to multiply force_matrix by diff / dist^3
            # force_matrix is 1/dist^2, so we need 1/dist^3 = force_matrix / dist
            # But we capped dist, so use dist_capped for the direction normalization
            norm = dist_capped[:, :, np.newaxis]
            force = np.sum(force_matrix[:, :, np.newaxis] * diff / norm, axis=1)
            
            # Add boundary attraction (keep particles inside the square)
            # Attract particles towards the center if they're too close to the boundary
            margin = 0.1
            boundary_force = np.zeros_like(positions)
            boundary_force[:, 0] += np.where(positions[:, 0] > 0.5 - margin, 1.0, 0.0) - \
                                   np.where(positions[:, 0] < -0.5 + margin, 1.0, 0.0)
            boundary_force[:, 1] += np.where(positions[:, 1] > 0.5 - margin, 1.0, 0.0) - \
                                   np.where(positions[:, 1] < -0.5 + margin, 1.0, 0.0)
            
            # Update positions
            positions += step_size * (force + 0.1 * boundary_force)
            
            # Clamp to be inside the square
            positions = np.clip(positions, -0.499, 0.499)
            
            # Decay step size
            step_size *= 0.999
            
            if step % 100 == 0 and time.time() > deadline - 10:
                break
        
        return positions
    
    best_positions = None
    best_radius = -np.inf
    
    # Try different initializations
    initializations = []
    
    # Square grid
    initializations.append(('square_grid', square_grid_initialization(N)))
    
    # Hexagonal
    initializations.append(('hexagonal', hexagonal_initialization(N)))
    
    # Random with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        if time.time() > deadline:
            break
        initializations.append((f'random_{seed}', random_initialization(N, seed=seed)))
    
    for init_name, init_positions in initializations:
        if time.time() > deadline:
            break
        
        # Optimize from this initialization
        try:
            optimized_positions = optimize_repulsion(init_positions)
            current_radius = compute_radius(optimized_positions)
            
            if current_radius > best_radius:
                best_radius = current_radius
                best_positions = optimized_positions.copy()
        except Exception:
            continue
    
    # If optimization didn't find anything better, fall back to grid
    if best_positions is None:
        best_positions = square_grid_initialization(N)
    
    # Final validation: ensure all positions are within bounds
    best_positions = np.clip(best_positions, -0.4999, 0.4999)
    
    return best_positions
# EVOLVE-BLOCK-END
