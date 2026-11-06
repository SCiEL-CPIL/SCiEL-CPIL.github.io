"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Construct centers of N=100 equal non-overlapping spheres in [-0.5, 0.5]^3
    maximizing the minimum radius. Uses a repulsion-based iterative algorithm
    with simulated annealing to find a good packing configuration.
    
    The radius is r = min(min_{i<j} |c_i - c_j|/2, min_i min_k (0.5 - |c_ik|)).
    """
    n = N
    
    def compute_r(centers):
        """Compute the radius given centers."""
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        x = centers[:, 0]
        y = centers[:, 1]
        z = centers[:, 2]
        boundary_dists = np.min(np.stack([
            0.5 - np.abs(x),
            0.5 - np.abs(y),
            0.5 - np.abs(z)
        ], axis=1), axis=1)
        min_boundary = np.min(boundary_dists)
        
        return min(min_pair_dist / 2.0, min_boundary)
    
    def repulsion_step(centers, step_size, rng):
        """
        Perform one repulsion step: push spheres away from their nearest neighbors.
        Returns updated centers.
        """
        new_centers = centers.copy()
        n_points = len(centers)
        
        # Compute all pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # For each sphere, find the closest neighbor and push away
        for i in range(n_points):
            # Find the closest neighbor
            min_dist_idx = np.argmin(dists[i])
            min_dist = dists[i, min_dist_idx]
            
            if min_dist < 1e-10:
                # If coincident, push randomly
                direction = rng.normal(size=3)
                direction /= (np.linalg.norm(direction) + 1e-10)
            else:
                # Push away from closest neighbor
                direction = (centers[i] - centers[min_dist_idx]) / min_dist
            
            # Also consider boundary: push inward if too close to boundary
            x, y, z = centers[i]
            boundary_push = np.zeros(3)
            
            # Distance to each face
            d_x_pos = 0.5 - x
            d_x_neg = 0.5 + x
            d_y_pos = 0.5 - y
            d_y_neg = 0.5 + y
            d_z_pos = 0.5 - z
            d_z_neg = 0.5 + z
            
            # If close to boundary, push inward
            if d_x_pos < 0.05:
                boundary_push[0] += 1.0
            if d_x_neg < 0.05:
                boundary_push[0] -= 1.0
            if d_y_pos < 0.05:
                boundary_push[1] += 1.0
            if d_y_neg < 0.05:
                boundary_push[1] -= 1.0
            if d_z_pos < 0.05:
                boundary_push[2] += 1.0
            if d_z_neg < 0.05:
                boundary_push[2] -= 1.0
            
            # Combine repulsion and boundary forces
            force = direction * 0.7 + boundary_push * 0.3
            force_norm = np.linalg.norm(force)
            if force_norm > 1e-10:
                force /= force_norm
            
            # Add small random perturbation to avoid symmetry
            force += rng.normal(scale=0.1, size=3)
            
            new_centers[i] += step_size * force
        
        # Enforce bounds
        new_centers = np.clip(new_centers, -0.499, 0.499)
        
        return new_centers
    
    def anneal(centers, n_iter=2000, rng=None):
        """
        Perform simulated annealing with repulsion steps.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        best_centers = centers.copy()
        best_r = compute_r(centers)
        current_r = best_r
        
        # Cooling schedule
        initial_step = 0.05
        final_step = 0.001
        cooling_rate = (final_step / initial_step) ** (1.0 / n_iter)
        
        for it in range(n_iter):
            step_size = initial_step * (cooling_rate ** it)
            
            # Perform repulsion step
            new_centers = repulsion_step(centers, step_size, rng)
            new_r = compute_r(new_centers)
            
            # Accept if better, or with some probability if worse (annealing)
            if new_r >= current_r:
                centers = new_centers
                current_r = new_r
                if new_r > best_r:
                    best_r = new_r
                    best_centers = new_centers.copy()
            else:
                # Accept worse solutions with decreasing probability
                delta = current_r - new_r
                accept_prob = np.exp(-delta / (step_size * 0.1))
                if rng.random() < accept_prob:
                    centers = new_centers
                    current_r = new_r
        
        return best_centers, best_r
    
    # Multi-start with different random seeds
    best_centers_overall = None
    best_r_overall = 0.0
    
    for seed in [42, 123, 456, 789, 1000]:
        rng = np.random.default_rng(seed)
        
        # Initialize with a slightly perturbed grid
        m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        if len(grid) >= n:
            centers_init = grid[:n].copy()
        else:
            extra = rng.uniform(-0.4, 0.4, size=(n - len(grid), 3))
            centers_init = np.vstack([grid, extra])
        
        # Add perturbation
        centers_init += rng.uniform(-0.02, 0.02, size=centers_init.shape)
        centers_init = np.clip(centers_init, -0.49, 0.49)
        
        # Run annealing
        centers_opt, r_opt = anneal(centers_init, n_iter=1500, rng=rng)
        
        if r_opt > best_r_overall:
            best_r_overall = r_opt
            best_centers_overall = centers_opt
    
    # If no good result, fallback to grid
    if best_centers_overall is None:
        m = int(np.ceil(n ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers_overall = grid[:n]
    
    # Final cleanup
    best_centers_overall = np.clip(best_centers_overall, -0.5, 0.5)
    
    return best_centers_overall
# EVOLVE-BLOCK-END
