"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Maximizes the radius of 50 equal spheres in a unit cube [-0.5, 0.5]^3.
    Uses Simulated Annealing with a cooling schedule.
    """
    n_spheres = 50
    dim = 3
    size = 0.5
    
    # Objective function: returns the negative of the minimum radius
    # r = min( min_{i<j} ||c_i - c_j|| / 2, min_i (0.5 - max(|c_i|)) )
    def get_radius(centers):
        # centers: (N, 3)
        # Check boundaries
        # Distance to nearest wall for each sphere
        wall_dists = size - np.max(np.abs(centers), axis=1)
        min_wall_dist = np.min(wall_dists)
        
        # Pairwise distances
        # Compute distance matrix
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        return r

    def objective(centers):
        # We want to maximize radius, so minimize -radius
        # Add a large penalty if any sphere goes out of bounds to keep SA stable
        # Although we will clip, it's better to penalize strongly in objective for SA acceptance
        if np.any(np.abs(centers) > size):
            return 1e10
        return -get_radius(centers)

    # Initialization: Random points inside the cube
    # To improve initial quality, we can use a jittered grid or just random.
    # Random is fine for SA if we run long enough.
    # Let's use a slightly better init: random but ensuring no immediate overlaps if possible?
    # No, just random uniform.
    
    np.random.seed(42)
    current_centers = np.random.uniform(-size + 0.01, size - 0.01, (n_spheres, dim))
    current_obj = objective(current_centers)
    
    best_centers = current_centers.copy()
    best_obj = current_obj
    
    # Simulated Annealing parameters
    start_temp = 0.1
    end_temp = 1e-6
    n_iterations = 20000 # Adjust based on time limit
    cooling_rate = (end_temp / start_temp) ** (1.0 / n_iterations)
    
    start_time = time.time()
    time_limit = 100.0 # Leave 20s buffer
    
    for i in range(n_iterations):
        if time.time() - start_time > time_limit:
            break
            
        temp = start_temp * (cooling_rate ** i)
        
        # Generate a new candidate
        # Perturb a random subset of points? Or all?
        # Perturbing all points with small noise is standard.
        # Step size should decrease with temperature or be fixed relative to scale.
        # Let's use a step size proportional to temperature to explore more at high T.
        step_size = 0.01 * (temp / start_temp) + 0.001
        noise = np.random.normal(0, step_size, (n_spheres, dim))
        
        new_centers = current_centers + noise
        
        # Clip to cube to ensure validity (and handle wall constraints implicitly via objective penalty if we didn't, 
        # but clipping ensures we stay in bounds. However, clipping changes the distance to wall.
        # If we clip, we are effectively projecting. This is valid as long as we evaluate the clipped position.
        # But if we just clip, we might get stuck at boundaries. 
        # Better: allow them to go out, but penalize heavily in objective.
        # The objective function above penalizes > 0.5.
        
        new_obj = objective(new_centers)
        
        # Acceptance criterion
        delta_obj = new_obj - current_obj
        if delta_obj < 0:
            # Improvement
            current_centers = new_centers
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_centers = current_centers.copy()
        else:
            # Worse, accept with probability exp(-delta_obj / temp)
            if np.random.random() < np.exp(-delta_obj / temp):
                current_centers = new_centers
                current_obj = new_obj

    # Final polish: local search on the best solution found
    # Use a very small step size local search (hill climbing)
    polish_steps = 1000
    polish_step = 0.001
    for _ in range(polish_steps):
        if time.time() - start_time > time_limit:
            break
        noise = np.random.normal(0, polish_step, (n_spheres, dim))
        new_centers = best_centers + noise
        new_obj = objective(new_centers)
        if new_obj < best_obj:
            best_centers = new_centers
            best_obj = new_obj
            
    # Ensure final result is strictly inside
    best_centers = np.clip(best_centers, -size + 1e-9, size - 1e-9)
    
    return best_centers
# EVOLVE-BLOCK-END
