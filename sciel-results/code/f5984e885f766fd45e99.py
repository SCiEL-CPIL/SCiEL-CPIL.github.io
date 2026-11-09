"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Constructs the centers of 300 circles inside a unit circle.
    Uses Simulated Annealing to maximize the minimum packing radius.
    """
    
    # 1. Initialization: Random points inside the unit circle
    # We use random uniform sampling in the square and reject outside points
    # Or simply generate random angles and radii.
    # Radius distribution should be sqrt(r) for uniform area density.
    radii = np.sqrt(np.random.uniform(0, 1, N))
    angles = np.random.uniform(0, 2 * np.pi, N)
    pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    # Ensure they are strictly inside (or on boundary)
    # The random generation ensures |pos| <= 1.
    
    def get_score(positions):
        """Calculate the packing radius r."""
        # Pairwise distances
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_pair_dist_sq = np.min(dists_sq)
        min_pair_dist = np.sqrt(min_pair_dist_sq)
        
        # Boundary distances
        radii = np.linalg.norm(positions, axis=1)
        boundary_dists = 1.0 - radii
        min_boundary_dist = np.min(boundary_dists)
        
        # The score is the min of the two constraints
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        return r

    current_score = get_score(pos)
    best_pos = pos.copy()
    best_score = current_score
    
    # 2. Simulated Annealing Parameters
    # Initial temperature should be high enough to accept bad moves
    # The score range is roughly 0 to ~0.05.
    T_start = 0.01
    T_end = 1e-6
    iterations = 50000
    # Cooling schedule: exponential
    cooling_rate = (T_end / T_start) ** (1.0 / iterations)
    
    # Perturbation scale
    # Initial scale should be comparable to the expected spacing
    # Spacing ~ sqrt(pi/300) ~ 0.1
    base_step = 0.05
    
    T = T_start
    
    for i in range(iterations):
        # Perturbation
        # Choose a small number of points to move
        # Moving 1 point is standard for SA.
        # Let's try moving 1 random point.
        idx = np.random.randint(N)
        
        # Gaussian perturbation
        # Scale the step size with temperature?
        # Or keep it constant and let T handle acceptance.
        # Often, step size is also annealed.
        step_scale = base_step * (T / T_start) ** 0.5 # Reduce step size as we cool
        
        noise = np.random.normal(0, step_scale, 2)
        
        new_pos = pos.copy()
        new_pos[idx] += noise
        
        # Constraint: Project back to unit circle if outside
        r_new_idx = np.linalg.norm(new_pos[idx])
        if r_new_idx > 1.0:
            new_pos[idx] = new_pos[idx] / r_new_idx
        
        new_score = get_score(new_pos)
        
        # Acceptance Criterion
        delta = new_score - current_score
        if delta > 0:
            # Better, always accept
            pos = new_pos
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pos = pos.copy()
        else:
            # Worse, accept with probability exp(delta / T)
            if T > 0:
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    pos = new_pos
                    current_score = new_score
        
        # Cool down
        T *= cooling_rate
        
        # Early stopping if T is too low or no improvement for a long time
        if T < T_end:
            break

    return best_pos

# EVOLVE-BLOCK-END
