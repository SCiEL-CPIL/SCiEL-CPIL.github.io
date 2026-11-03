import numpy as np
import time
import random

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin

    def compute_radius(centers):
        """Compute the radius for a given configuration."""
        if centers is None or len(centers) != N:
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        center_dists = np.sqrt(np.sum(centers**2, axis=1))
        min_boundary_dist = np.min(1.0 - center_dists)
        
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        return r if r > 0 else 0.0

    def initialize_rings():
        """
        Initialize points in concentric rings based on known packing structure.
        This provides a strong starting point close to the optimal topology.
        """
        points = np.zeros((N, 2))
        
        # Structure: 1 center, 6, 12, 18, 13 (Total 50)
        # This is a common topology for N=50 packings.
        
        # Center
        points[0] = [0.0, 0.0]
        
        # Ring 1: 6 points
        r1 = 0.25
        for i in range(6):
            angle = 2 * np.pi * i / 6
            points[1 + i] = [r1 * np.cos(angle), r1 * np.sin(angle)]
            
        # Ring 2: 12 points
        r2 = 0.45
        for i in range(12):
            angle = 2 * np.pi * i / 12
            points[7 + i] = [r2 * np.cos(angle), r2 * np.sin(angle)]
            
        # Ring 3: 18 points
        r3 = 0.65
        for i in range(18):
            angle = 2 * np.pi * i / 18
            points[19 + i] = [r3 * np.cos(angle), r3 * np.sin(angle)]
            
        # Ring 4: 13 points
        r4 = 0.85
        for i in range(13):
            angle = 2 * np.pi * i / 13
            points[37 + i] = [r4 * np.cos(angle), r4 * np.sin(angle)]
            
        return points

    def energy(centers):
        """
        Calculate the objective function.
        We want to maximize the minimum distance.
        Equivalent to minimizing the negative of the minimum distance.
        To handle hard constraints, we add a huge penalty if any constraint is violated.
        
        Objective: 
        1. All centers must be inside the unit circle.
        2. All pairwise distances must be >= 2 * r_target? 
           Actually, we are maximizing r. 
           Let's define the score as:
           r = min(min_pair_dist/2, min_boundary_dist)
           
           In Simulated Annealing, we usually minimize an energy.
           Let Energy = -r + Penalty.
           
           Penalty:
           If any center is outside, Penalty is huge.
           If any pair is closer than 2*r_current? No, r is derived from the positions.
           
           Better approach for SA:
           We treat the problem as: Find positions that maximize the minimum clearance.
           Energy = -min_clearance.
           
           However, standard SA works better with a smooth landscape or a landscape where 
           "bad" states are clearly worse.
           
           Let's use:
           E = -min_pair_dist/2  (if valid)
           If invalid (outside), E = 1000.
           
           But this has a problem: if points overlap, min_pair_dist is small, so -min_pair_dist/2 is large (bad).
           This naturally penalizes overlaps.
           
           We also need to penalize being outside the boundary.
           boundary_margin = 1 - max_norm.
           If boundary_margin < 0, it's invalid.
           
           Let's define:
           min_pair = min pairwise distance
           min_bound = 1 - max norm
           
           r_candidate = min(min_pair / 2, min_bound)
           
           If min_bound < 0:
               E = 1000 - min_bound (penalize going out)
           Else:
               E = -r_candidate
               
           This is a step function at the boundary, but smooth otherwise.
           """
        norms = np.linalg.norm(centers, axis=1)
        max_norm = np.max(norms)
        min_bound = 1.0 - max_norm
        
        if min_bound < 0:
            # Hard penalty for being outside
            return 1000.0 - min_bound
        
        # Compute min pairwise distance
        # Vectorized
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        r = min(min_pair / 2.0, min_bound)
        
        return -r

    def propose_move(centers, step_size):
        """
        Generate a new candidate configuration.
        Mix of:
        1. Small Gaussian moves (local search)
        2. Swapping two points (topology change)
        3. Rotating a ring (symmetry breaking)
        """
        new_centers = centers.copy()
        move_type = random.random()
        
        if move_type < 0.7:
            # Local move: pick a random point and nudge it
            i = random.randint(0, N - 1)
            new_centers[i] += np.random.normal(0, step_size, 2)
        elif move_type < 0.9:
            # Swap: pick two random points and swap their positions
            i, j = random.sample(range(N), 2)
            new_centers[i], new_centers[j] = new_centers[j].copy(), new_centers[i].copy()
        else:
            # Ring rotation: This is tricky without knowing which points belong to which ring.
            # Instead, do a "kick" to a random subset of points to allow larger rearrangements.
            subset_size = random.randint(2, 5)
            indices = random.sample(range(N), subset_size)
            for i in indices:
                new_centers[i] += np.random.normal(0, step_size * 2, 2)
                
        return new_centers

    # Initial state
    centers = initialize_rings()
    
    # Add small jitter to break exact symmetry
    centers += np.random.normal(0, 0.01, centers.shape)
    
    # Ensure initial validity
    norms = np.linalg.norm(centers, axis=1)
    for i in range(N):
        if norms[i] > 0.99:
            centers[i] *= 0.99 / norms[i]
            
    best_centers = centers.copy()
    best_energy = energy(centers)
    best_radius = compute_radius(centers)
    
    current_energy = best_energy
    current_centers = best_centers.copy()
    
    # Simulated Annealing Parameters
    T_start = 0.1
    T_end = 0.001
    # We don't know the exact number of steps, so we run until deadline.
    # Estimate steps: assume ~1000 steps per second? 
    # Let's just run a loop and check time.
    
    step_size = 0.05
    alpha = 0.9999 # Cooling factor per step
    
    while time.time() < deadline:
        T = max(T_start * (T_end / T_start) ** (0.001), T_end) 
        # A better cooling schedule is geometric, but since we don't know N_steps, 
        # let's use a fixed schedule that decays over time.
        elapsed = time.time() - start_time
        # Linear cooling from T_start to T_end over 100s
        T = T_start - (T_start - T_end) * (elapsed / 100.0)
        if T < T_end:
            T = T_end
            
        # Adaptive step size: shrink as T drops
        step_size = 0.05 * (T / T_start) + 0.001
        
        new_centers = propose_move(current_centers, step_size)
        
        # Check hard constraints for new state
        new_norms = np.linalg.norm(new_centers, axis=1)
        if np.any(new_norms > 1.0):
             # Reject immediately if outside, or apply penalty?
             # Penalty is handled in energy, but huge penalty makes acceptance unlikely.
             pass
             
        new_energy = energy(new_centers)
        
        # Acceptance criterion
        if new_energy < current_energy:
            current_energy = new_energy
            current_centers = new_centers
            # Check if this is a new best
            new_radius = compute_radius(new_centers)
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
        else:
            # Metropolis criterion
            delta_E = new_energy - current_energy
            if delta_E < 0: # Should not happen due to above, but safe
                current_energy = new_energy
                current_centers = new_centers
                new_radius = compute_radius(new_centers)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = new_centers.copy()
            else:
                # Probability of accepting worse state
                # Avoid overflow in exp
                if delta_E / T < 700: 
                    prob = np.exp(-delta_E / T)
                    if random.random() < prob:
                        current_energy = new_energy
                        current_centers = new_centers
                        # Even if worse than best, it might lead to better later
                        # But we only track best_radius for return
                        pass
                # else: prob is effectively 0, reject

    # Final cleanup: ensure returned centers are strictly valid
    # Clip any points slightly outside due to floating point errors in SA
    final_centers = best_centers.copy()
    norms = np.linalg.norm(final_centers, axis=1)
    for i in range(N):
        if norms[i] > 1.0:
            final_centers[i] *= 0.9999 / norms[i]
            
    # Verify final radius
    final_radius = compute_radius(final_centers)
    if final_radius <= 0:
        # Fallback to initial if something went wrong
        final_centers = initialize_rings()
        
    return final_centers
# EVOLVE-BLOCK-END
