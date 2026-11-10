import numpy as np
import time
from scipy.spatial.distance import cdist

N = 100

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin for safety
    
    best_r = 0.0
    best_centers = None
    
    def get_min_radius(centers):
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)

    def optimize(centers, max_steps=50000, step_size=0.005, repulsion_strength=1.0):
        """Force-directed relaxation with soft repulsion from all neighbors."""
        centers = centers.copy()
        current_r = get_min_radius(centers)
        best_local_r = current_r
        best_local_centers = centers.copy()
        
        # Adaptive step size
        step = step_size
        no_improve_count = 0
        max_no_improve = 2000
        
        for i in range(max_steps):
            if time.time() - start_time > time_limit:
                break
            
            # Compute all pairwise distances
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            
            # Compute repulsion forces from all neighbors
            # Force magnitude: repulsion_strength / dist^2 (capped to avoid blowup)
            # Direction: away from neighbor
            forces = np.zeros_like(centers)
            
            # Soft repulsion: f = strength / (dist^2 + epsilon)
            # This is smoother than hard repulsion
            inv_dist_sq = 1.0 / (dist**2 + 1e-10)
            inv_dist_sq = np.where(np.eye(N, dtype=bool), 0.0, inv_dist_sq)
            inv_dist_sq = np.minimum(inv_dist_sq, 100.0)  # Cap to prevent explosion
            
            # Force from each neighbor: direction * magnitude
            # forces[i] = sum_j (centers[i] - centers[j]) / dist[i,j]^2
            # This is: for each i, sum over j of (centers[i] - centers[j]) * inv_dist_sq[i,j]
            
            # Vectorized: forces[i] = sum_j (centers[i] - centers[j]) * inv_dist_sq[i,j]
            # = centers[i] * sum_j inv_dist_sq[i,j] - sum_j centers[j] * inv_dist_sq[i,j]
            
            row_sums = np.sum(inv_dist_sq, axis=1)  # (N,)
            weighted_centers = inv_dist_sq @ centers  # (N, 3)
            
            forces = centers * row_sums[:, np.newaxis] - weighted_centers
            forces *= repulsion_strength
            
            # Wall repulsion: push points away from walls
            # For each coordinate, if center is close to wall, push inward
            for k in range(3):
                # Distance to positive wall: 0.5 - centers[:, k]
                # Distance to negative wall: 0.5 + centers[:, k]
                dist_to_pos_wall = 0.5 - centers[:, k]
                dist_to_neg_wall = 0.5 + centers[:, k]
                
                # Repulsion from walls: strength / dist^2
                # Only active when close to wall
                wall_force = np.zeros(N)
                
                # Positive wall: push in negative direction if close
                close_to_pos = dist_to_pos_wall < 0.1
                if np.any(close_to_pos):
                    wall_force[close_to_pos] += repulsion_strength / (dist_to_pos_wall[close_to_pos]**2 + 1e-10)
                    wall_force[close_to_pos] = np.minimum(wall_force[close_to_pos], 100.0)
                
                # Negative wall: push in positive direction if close
                close_to_neg = dist_to_neg_wall < 0.1
                if np.any(close_to_neg):
                    wall_force[close_to_neg] -= repulsion_strength / (dist_to_neg_wall[close_to_neg]**2 + 1e-10)
                    wall_force[close_to_neg] = np.abs(wall_force[close_to_neg])
                
                forces[:, k] += wall_force
            
            # Normalize forces to prevent large steps
            force_magnitudes = np.linalg.norm(forces, axis=1)
            max_force = np.max(force_magnitudes) if np.max(force_magnitudes) > 0 else 1.0
            if max_force > 1.0:
                forces /= max_force
            
            # Update positions
            centers += step * forces
            
            # Clamp to box (with small margin to avoid being exactly on boundary)
            centers = np.clip(centers, -0.499, 0.499)
            
            # Compute new radius
            new_r = get_min_radius(centers)
            
            # Adaptive step size
            if new_r > current_r:
                step = min(step * 1.1, 0.02)
                no_improve_count = 0
                if new_r > best_local_r:
                    best_local_r = new_r
                    best_local_centers = centers.copy()
            else:
                step = max(step * 0.9, 0.001)
                no_improve_count += 1
            
            current_r = new_r
            
            # If stuck for too long, add random perturbation
            if no_improve_count > max_no_improve:
                # Perturb a random subset of points
                n_perturb = max(1, N // 10)
                perturbed = np.random.choice(N, n_perturb, replace=False)
                centers[perturbed] += np.random.randn(n_perturb, 3) * 0.01
                centers = np.clip(centers, -0.499, 0.499)
                no_improve_count = 0
                step = 0.005
        
        return best_local_centers, best_local_r

    # Initial Configurations
    
    # 1. Cubic Grid (Seed)
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    init_grid = grid[:N]
    
    c1, r1 = optimize(init_grid, max_steps=30000)
    if r1 > best_r:
        best_r = r1
        best_centers = c1

    # 2. FCC-like Initialization
    s = 0.3
    pts = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                base = np.array([i, j, k]) * s
                pts.append(base)
                pts.append(base + np.array([s/2, s/2, 0]))
                pts.append(base + np.array([s/2, 0, s/2]))
                pts.append(base + np.array([0, s/2, s/2]))
    
    fcc_grid = np.array(pts)
    fcc_grid -= fcc_grid.mean(axis=0)
    extent = np.max(np.abs(fcc_grid))
    if extent > 0.5:
        fcc_grid *= (0.5 / extent)
    
    dists_from_origin = np.linalg.norm(fcc_grid, axis=1)
    order = np.argsort(dists_from_origin)
    init_fcc = fcc_grid[order[:N]]
    
    c2, r2 = optimize(init_fcc, max_steps=30000)
    if r2 > best_r:
        best_r = r2
        best_centers = c2

    # 3. Random Initialization
    np.random.seed(42)
    init_random = np.random.uniform(-0.4, 0.4, (N, 3))
    c3, r3 = optimize(init_random, max_steps=30000)
    if r3 > best_r:
        best_r = r3
        best_centers = c3

    # 4. If time remains, further optimize the best configuration
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 15 and best_centers is not None:
        c_best, r_best = optimize(best_centers, max_steps=20000, step_size=0.003)
        if r_best > best_r:
            best_r = r_best
            best_centers = c_best

    if best_centers is None:
        return init_grid

    return best_centers
