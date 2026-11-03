import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Uses Simulated Annealing with vectorized force calculations and binary search
    on the radius to find a near-optimal packing of N circles in a unit circle.
    """
    
    def compute_energy_and_forces(points, r):
        """
        Computes the total energy (sum of squared overlaps) and the gradient (forces).
        Energy is defined as:
        E = sum_{i<j} max(0, 2r - dist_ij)^2 + sum_i max(0, dist_i - (1-r))^2
        Forces are the negative gradient of E.
        """
        # Pairwise distances
        # diff[i, j, :] = points[i] - points[j]
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Mask for i < j to avoid double counting and self-interaction
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        # Overlaps for pairs
        # overlap_ij = max(0, 2r - dist_ij)
        overlaps = np.maximum(0.0, 2 * r - dists)
        overlaps[~mask] = 0.0
        
        # Energy contribution from pairs
        pair_energy = np.sum(overlaps ** 2)
        
        # Force contribution from pairs
        # dE/dx_i = -2 * sum_{j!=i} overlap_ij * (x_i - x_j) / dist_ij
        # Note: overlap is only non-zero if dist < 2r.
        # Gradient of max(0, 2r - d)^2 w.r.t d is -2(2r-d).
        # Gradient of d w.r.t x_i is (x_i - x_j)/d.
        # So force on i from j is -dE/dx_i = 2 * overlap_ij * (x_i - x_j) / dist_ij
        
        # To avoid division by zero, where dist is 0, we can treat force as 0 or random.
        # Since we initialize with distinct points and use SA, exact 0 is rare.
        # We use np.divide with where clause.
        
        # We need the vector (x_i - x_j) / dist_ij for all i, j
        # This is equivalent to diff / dists[:, :, np.newaxis]
        
        # Handle dists == 0
        safe_dists = np.where(dists > 1e-9, dists, 1.0)
        unit_vectors = diff / safe_dists[:, :, np.newaxis]
        
        # Force from pair j on i:
        # F_ij = 2 * overlap_ij * unit_vector_ij
        # Total force on i = sum_j F_ij
        # Note: overlap is symmetric.
        # We sum over j for each i.
        # forces[i] = sum_j 2 * overlap_ij * unit_vector_ij
        
        # We can compute this as:
        # forces = 2 * np.sum(overlaps[:, :, np.newaxis] * unit_vectors, axis=1)
        # But we must ensure we only sum over j where mask is true? 
        # Actually, overlaps[~mask] is 0, so it's fine.
        # However, overlaps[i,i] is 0 because dist=0 -> overlap = 2r, but mask excludes diagonal?
        # Wait, mask is k=1, so diagonal is False. overlaps[i,i] becomes 0.
        # So the sum includes j=i term which is 0. Correct.
        
        pair_forces = 2.0 * np.sum(overlaps[:, :, np.newaxis] * unit_vectors, axis=1)
        
        # Boundary overlaps
        # dist_i = |points[i]|
        dists_to_origin = np.linalg.norm(points, axis=1)
        boundary_overlaps = np.maximum(0.0, dists_to_origin - (1.0 - r))
        
        boundary_energy = np.sum(boundary_overlaps ** 2)
        
        # Force from boundary
        # E_b = (d_i - (1-r))^2
        # dE_b/dx_i = 2 * (d_i - (1-r)) * (x_i / d_i)
        # Force = -dE/dx = -2 * boundary_overlap_i * (x_i / d_i)
        
        safe_dists_origin = np.where(dists_to_origin > 1e-9, dists_to_origin, 1.0)
        boundary_forces = -2.0 * boundary_overlaps[:, np.newaxis] * (points / safe_dists_origin[:, np.newaxis])
        
        total_energy = pair_energy + boundary_energy
        total_forces = pair_forces + boundary_forces
        
        return total_energy, total_forces

    def simulated_annealing(points, r, max_iters=2000, T_start=0.1, T_end=1e-5, alpha=0.995):
        """
        Runs Simulated Annealing to minimize energy for a fixed radius r.
        """
        current_points = points.copy()
        current_energy, _ = compute_energy_and_forces(current_points, r)
        
        best_points = current_points.copy()
        best_energy = current_energy
        
        for i in range(max_iters):
            T = T_start * (T_end / T_start) ** (i / max_iters)
            
            # Compute forces (gradient)
            _, forces = compute_energy_and_forces(current_points, r)
            
            # Move in direction of negative gradient with some noise
            # Step size proportional to T
            step_size = 0.1 * T
            noise = np.random.normal(0, step_size, size=(N, 2))
            proposed_points = current_points - forces * 0.01 + noise
            
            # Ensure points stay within a reasonable bound (e.g., radius 1.5)
            # If they go too far, project back or reject
            proposed_norms = np.linalg.norm(proposed_points, axis=1)
            # We don't strictly need to project if energy handles it, 
            # but it helps stability.
            
            proposed_energy, _ = compute_energy_and_forces(proposed_points, r)
            
            # Acceptance probability
            if proposed_energy < current_energy:
                current_energy = proposed_energy
                current_points = proposed_points
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_points = current_points.copy()
            else:
                # Metropolis criterion
                delta_e = proposed_energy - current_energy
                if delta_e < 1e-12:
                    continue
                prob = np.exp(-delta_e / T)
                if np.random.rand() < prob:
                    current_energy = proposed_energy
                    current_points = proposed_points
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_points = current_points.copy()
                        
        return best_points, best_energy

    def is_feasible(points, r, tol=1e-4):
        """
        Check if the configuration is valid (no overlaps, inside circle).
        Returns True if max overlap is below tol.
        """
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        
        dists_to_origin = np.linalg.norm(points, axis=1)
        max_radius = np.max(dists_to_origin)
        
        pair_violation = max(0, 2 * r - min_pairwise)
        boundary_violation = max(0, max_radius - (1 - r))
        
        return max(pair_violation, boundary_violation) < tol

    # Initialization
    # Use a spiral initialization which is often a good starting point
    points_init = np.zeros((N, 2))
    for i in range(N):
        r_init = np.sqrt(i / N) * 0.9
        theta = i * np.pi * (3 - np.sqrt(5))
        points_init[i] = [r_init * np.cos(theta), r_init * np.sin(theta)]
    
    start_time = time.time()
    time_limit = 110.0 # Leave some margin
    
    # Binary search for the optimal radius
    low_r = 0.1
    high_r = 0.13
    
    # First, verify low_r is feasible with a quick check
    # If not, we might need to adjust, but 0.1 is definitely feasible for N=50
    # 50 * pi * 0.1^2 = 0.157 * pi ~ 0.5 < pi (area of unit circle). So area-wise it's fine.
    
    best_points = points_init
    best_r = low_r
    
    # We will do a few iterations of binary search
    # Since SA is stochastic, we run it multiple times for a given r to be confident.
    
    while time.time() - start_time < time_limit:
        mid_r = (low_r + high_r) / 2
        
        # Run SA for mid_r
        # We use the current best_points as a warm start if possible, 
        # or random perturbation of init.
        # Using best_points from lower r as start is a good heuristic.
        start_points = best_points.copy()
        
        # Add some noise to escape local minima
        start_points += np.random.normal(0, 0.01, size=(N, 2))
        
        # Run SA
        # Adjust max_iters based on remaining time
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 5:
            break
            
        # Estimate time per SA run. Let's assume ~1s per 1000 iters?
        # Vectorized ops are fast. Let's try 1000 iters.
        max_iters = 1000
        
        points_res, energy_res = simulated_annealing(start_points, mid_r, max_iters=max_iters)
        
        # Check feasibility
        # We use a strict check on the final best points found by SA
        if is_feasible(points_res, mid_r, tol=1e-5):
            # Feasible, try larger radius
            low_r = mid_r
            best_r = mid_r
            best_points = points_res
        else:
            # Not feasible, try smaller radius
            high_r = mid_r
            
        # Convergence check
        if high_r - low_r < 1e-5:
            break

    # Final Polish: Run SA again on the best found radius with more iterations
    # to squeeze out a bit more accuracy
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 2:
        # Use best_points as start
        start_points = best_points.copy()
        # Small noise
        start_points += np.random.normal(0, 0.005, size=(N, 2))
        
        # Run for as many iterations as possible
        # Estimate iters: ~0.5ms per iter? So 10000 iters in 5s.
        max_iters_final = int(remaining_time * 2000) 
        if max_iters_final < 100:
            max_iters_final = 100
            
        points_final, energy_final = simulated_annealing(start_points, best_r, max_iters=max_iters_final, T_start=0.05, T_end=1e-6, alpha=0.999)
        
        if is_feasible(points_final, best_r, tol=1e-5):
            best_points = points_final
        else:
            # If final polish failed, keep the previous best
            pass

    # Final validation and cleanup
    # Calculate the actual score
    def get_score(points):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        radii = np.linalg.norm(points, axis=1)
        min_boundary = 1.0 - np.max(radii)
        return min(min_pairwise / 2.0, min_boundary)

    final_score = get_score(best_points)
    
    # Ensure validity: if score is negative or very small, scale down
    # If boundary is violated, scale towards center
    radii = np.linalg.norm(best_points, axis=1)
    if np.any(radii > 1.0 - final_score + 1e-9):
        scale = (1.0 - final_score - 1e-9) / np.max(radii)
        best_points *= scale
        final_score = get_score(best_points)
        
    # If pairwise is violated (score determined by pairwise), we can't easily fix without moving points.
    # But SA should have minimized this.
    
    return best_points

# EVOLVE-BLOCK-END
