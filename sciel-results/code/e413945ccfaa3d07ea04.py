"""
Strategy: Stochastic Local Search via Simulated Annealing
This approach abandons the continuous differentiable optimization (smooth min)
used in the previous strategy. Instead, it treats the problem as a discrete
combinatorial optimization task.

Method:
1. Initialize with a random or grid-based configuration.
2. Perform Simulated Annealing:
   - At each step, select a random sphere and apply a random perturbation (Gaussian noise).
   - Compute the exact radius (min distance to neighbors/walls).
   - Accept the move if it improves the radius, or with probability exp(-delta/T) if it degrades it, where T is the current temperature.
   - Decrease temperature T over time.
3. This allows the search to escape local minima by accepting worse moves early on,
   then converging to a high-quality local maximum as T approaches 0.

Advantages over Smooth Min:
- Directly optimizes the true objective function (no surrogate approximation error).
- Robust to the non-differentiable nature of the min function.
- Can explore the configuration space more broadly than gradient ascent.
"""

import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Initialization
    # Start with a perturbed grid to ensure a reasonable starting density
    rng = np.random.default_rng(12345)
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add significant noise to break symmetry and allow SA to explore
    centers += rng.normal(0, 0.05, size=centers.shape)
    centers = np.clip(centers, -0.49, 0.49)

    def compute_exact_radius(c):
        """Compute the true radius for evaluation."""
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Wall clearance
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    best_centers = centers.copy()
    best_radius = compute_exact_radius(best_centers)
    
    current_centers = centers.copy()
    current_radius = best_radius

    # Simulated Annealing Parameters
    # Initial temperature should be comparable to typical radius changes
    T_start = 0.01
    T_end = 1e-6
    # Number of steps: we will run as many as time permits
    # We estimate steps per second to adjust if needed, but fixed logic is safer for reproducibility
    # Let's aim for ~100,000 steps if possible, but check time every 1000 steps.
    
    # To make SA effective, the perturbation scale should also anneal
    # Start with larger moves, end with smaller moves
    sigma_start = 0.05
    sigma_end = 0.001

    step_count = 0
    max_steps = 200000  # Hard cap, but time is the real limit

    for step in range(max_steps):
        if time.time() > deadline - 5.0:
            break
            
        # Annealing schedule (Exponential cooling)
        # progress from 0 to 1
        # We don't know total steps in advance due to time limit, so we use a fixed total for the schedule
        # Or we can use a dynamic schedule. Let's use a fixed total for the schedule logic
        # and just stop when time runs out.
        total_steps_for_schedule = 100000
        progress = min(step / total_steps_for_schedule, 1.0)
        
        T = T_start * (T_end / T_start) ** progress
        sigma = sigma_start * (sigma_end / sigma_start) ** progress
        
        # Select a random sphere
        idx = rng.integers(0, N)
        
        # Save old position
        old_pos = current_centers[idx].copy()
        
        # Generate new position
        # Gaussian perturbation
        new_pos = old_pos + rng.normal(0, sigma, size=3)
        
        # Clip to valid range (strictly inside to avoid boundary issues, 
        # but the radius calculation handles boundaries. 
        # However, if we clip, we might bias towards center. 
        # Better to allow any position and let radius calc penalize outside.
        # But the problem states "inside the cube". 
        # If a sphere goes outside, radius becomes negative or invalid?
        # Score definition: min(1/2 - |c_ik|). If |c| > 0.5, this is negative.
        # We want to maximize radius. Negative radius is bad.
        # So we can just let it go outside, the penalty will be huge.
        # However, for efficiency, clipping to [-0.5, 0.5] is standard.
        # Let's clip to [-0.5, 0.5] to keep it physically meaningful.
        new_pos = np.clip(new_pos, -0.5, 0.5)
        
        # Apply move
        current_centers[idx] = new_pos
        
        # Compute new radius
        # Optimization: We only need to recompute distances involving idx.
        # But for simplicity and correctness first, let's do full recompute.
        # N=100, N^2 = 10,000 ops. This is fast in NumPy.
        new_radius = compute_exact_radius(current_centers)
        
        # Acceptance criterion
        delta = new_radius - current_radius
        
        if delta > 0 or np.exp(-delta / T) > rng.random():
            # Accept move
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = current_centers.copy()
        else:
            # Reject move, revert
            current_centers[idx] = old_pos

    # Final validation
    final_radius = compute_exact_radius(best_centers)
    
    # Ensure no invalid states
    if np.any(np.isnan(best_centers)) or np.any(np.isinf(best_centers)):
        return centers # Fallback to initial grid
        
    if np.any(best_centers > 0.5) or np.any(best_centers < -0.5):
        best_centers = np.clip(best_centers, -0.5, 0.5)
        # Re-check radius after clip
        final_radius = compute_exact_radius(best_centers)
        
    return best_centers

# Test block (removed for submission)
if __name__ == "__main__":
    # c = construct()
    # print("Radius:", compute_exact_radius(c))
    pass
