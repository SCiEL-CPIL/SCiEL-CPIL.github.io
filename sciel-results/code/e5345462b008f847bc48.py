"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Use a deterministic seed for reproducibility
    rng = np.random.default_rng(42)
    
    # Initialize centers randomly within the cube
    # Add a small margin to start strictly inside
    margin = 1e-3
    centers = rng.uniform(-0.5 + margin, 0.5 - margin, (N, 3))
    
    # Simulated Annealing parameters
    # We want to maximize the minimum distance between centers and distance to walls.
    # Objective function to minimize: -min_dist
    # Or use a potential energy that penalizes overlaps heavily and rewards separation.
    
    # Let's define a score function:
    # score = min( min_pairwise_dist, min_wall_dist )
    # We want to maximize this.
    # In SA, we minimize energy. Let E = -score.
    # However, simple max-min is non-differentiable and noisy.
    # Better: Use a potential that is high when spheres are close (overlap) and lower when separated.
    # But we specifically want to maximize the *minimum* distance.
    # A common trick is to use a "soft" minimum or just directly optimize the min distance with SA.
    
    def get_min_distance(centers):
        # Calculate pairwise distances
        # Use broadcasting for efficiency
        # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=2)
        # np.fill_diagonal(dists, np.inf)
        # min_pair = np.min(dists)
        
        # More memory efficient for N=50:
        min_pair = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_pair:
                    min_pair = d
        
        # Calculate distance to walls
        # min_wall = min( min(centers[:,k] + 0.5), min(0.5 - centers[:,k]) )
        min_wall = np.inf
        for k in range(3):
            min_wall = min(min_wall, np.min(centers[:, k] + 0.5), np.min(0.5 - centers[:, k]))
            
        return min(min_pair, min_wall)
        
    # SA Loop
    T = 0.01
    T_min = 1e-6
    cooling = 0.995
    n_iter = 5000
    
    current_score = get_min_distance(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    for i in range(n_iter):
        T *= cooling
        if T < T_min:
            break
            
        # Propose a new state
        # Move one random sphere by a small amount
        idx = rng.integers(0, N)
        displacement = rng.normal(0, 0.01, 3)
        
        old_pos = centers[idx].copy()
        new_pos = old_pos + displacement
        centers[idx] = new_pos
        
        # Check if inside cube (hard constraint)
        if np.any(new_pos < -0.5) or np.any(new_pos > 0.5):
            centers[idx] = old_pos
            continue
            
        new_score = get_min_distance(centers)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_centers = centers.copy()
        else:
            # Reject
            centers[idx] = old_pos
            
    # Final cleanup: ensure strict interior
    # The score calculation handles the boundary, but we must return valid centers.
    # If a center is exactly on the boundary, radius is 0. We clip slightly inside.
    best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_centers
# EVOLVE-BLOCK-END
