"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Simulated Annealing Approach
    # Objective: Maximize r = min(min_pair_dist/2, min_bound_dist)
    
    # 1. Initialization
    np.random.seed(42)
    points = np.random.uniform(-0.5, 0.5, (N, 2))
    
    # 2. SA Parameters
    T0 = 0.05
    T_end = 0.00001
    cooling_rate = 0.999
    max_iter = 100000
    
    # Helper to calculate radius
    def calc_radius(pts):
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist) / 2.0
        
        # Boundary distances
        dx = 0.5 - np.abs(pts[:, 0])
        dy = 0.5 - np.abs(pts[:, 1])
        min_bound = np.min(np.minimum(dx, dy))
        
        return min(min_pair, min_bound)
    
    # Initial energy (negative radius)
    current_r = calc_radius(points)
    current_energy = -current_r
    best_r = current_r
    best_points = points.copy()
    
    T = T0
    
    for i in range(max_iter):
        # Pick a random number of points to move (1 to 5)
        n_move = np.random.randint(1, 6)
        indices = np.random.choice(N, n_move, replace=False)
        
        # Save original positions
        old_pos = points[indices].copy()
        
        # Generate new positions
        # Step size proportional to T
        step_size = T * 0.5
        new_pos = old_pos + np.random.normal(0, step_size, (n_move, 2))
        
        # Clip to bounds to ensure validity
        new_pos[:, 0] = np.clip(new_pos[:, 0], -0.5, 0.5)
        new_pos[:, 1] = np.clip(new_pos[:, 1], -0.5, 0.5)
        
        # Update points
        points[indices] = new_pos
        
        # Calculate new energy
        new_r = calc_radius(points)
        new_energy = -new_r
        
        # Delta energy
        delta_E = new_energy - current_energy
        
        # Metropolis acceptance criterion
        if delta_E < 0:
            # Improvement
            accept = True
        else:
            # Worse, accept with probability exp(-delta_E / T)
            if T > 0:
                accept = np.random.rand() < np.exp(-delta_E / T)
            else:
                accept = False
        
        if accept:
            current_energy = new_energy
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_points = points.copy()
        else:
            # Reject, revert
            points[indices] = old_pos
            
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end

    return best_points
# EVOLVE-BLOCK-END
