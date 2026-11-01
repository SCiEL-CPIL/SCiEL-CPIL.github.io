import numpy as np
import time
from scipy.spatial import cKDTree

N = 100

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We stop at 145s.
    deadline = start_time + 145.0 

    def compute_radius(c):
        """Compute the maximum common radius for given centers."""
        if not np.all(np.isfinite(c)):
            return 0.0
        if np.any(c > 0.5) or np.any(c < -0.5):
            return 0.0
        # Wall distance
        wall_dists = 0.5 - np.abs(c)
        r_wall = np.min(wall_dists)
        
        # Pairwise distance using cKDTree for efficiency
        # We need the nearest neighbor for each point.
        # cKDTree.query returns distances to k nearest neighbors.
        # k=2 gives self (0) and nearest neighbor.
        try:
            tree = cKDTree(c)
            dists, _ = tree.query(c, k=2)
            r_pair = np.min(dists[:, 1]) / 2.0
        except Exception:
            # Fallback if tree fails (e.g., duplicate points causing issues, though unlikely with clip)
            r_pair = 0.0
            
        return min(r_wall, r_pair)

    def force_directed_refine(centers, time_limit, step_scale=1.0):
        """
        Force-directed local search to maximize the minimum radius.
        Applies repulsive forces between overlapping spheres and attractive
        forces (or rather, corrective forces) for spheres too close to walls.
        """
        centers = centers.copy()
        r = compute_radius(centers)
        best_centers = centers.copy()
        best_r = r
        
        # Adaptive parameters
        base_step = 0.1 * step_scale
        decay_rate = 0.995
        
        it = 0
        while time.time() < time_limit:
            it += 1
            
            # Compute current radius and identify constraints
            wall_dists = 0.5 - np.abs(centers)
            r_wall = np.min(wall_dists)
            
            try:
                tree = cKDTree(centers)
                dists, _ = tree.query(centers, k=2)
                min_pair_dist = np.min(dists[:, 1])
                r_pair = min_pair_dist / 2.0
            except Exception:
                break
                
            current_r = min(r_wall, r_pair)
            
            # Stagnation check
            if it > 50 and current_r <= best_r * 1.00001:
                break
            
            # Compute forces
            force = np.zeros_like(centers)
            
            # 1. Wall constraints: Push spheres away from walls if they are too close
            # We want to increase the minimum wall distance.
            # If a sphere is closer to a wall than the current target radius, push it in.
            # Target radius is slightly above current r to create pressure.
            target_r = current_r + 0.01 
            
            for k in range(3):
                d_wall = 0.5 - np.abs(centers[:, k])
                # Spheres that are too close to the wall
                mask = d_wall < target_r
                if np.any(mask):
                    # Magnitude proportional to how much they violate the target
                    viol = target_r - d_wall[mask]
                    # Direction: towards the center (opposite to the sign of the coordinate)
                    # If x > 0, push left (-1). If x < 0, push right (+1).
                    dirs = -np.sign(centers[mask, k])
                    # Handle case where center is exactly 0
                    dirs[dirs == 0] = 1.0 
                    force[mask, k] += viol * dirs
            
            # 2. Pair constraints: Push apart spheres that are too close
            target_dist = 2 * target_r
            # Get all pairs within target distance
            # Using query_pairs is efficient for dense clusters
            pairs = tree.query_pairs(target_dist)
            
            for i, j in pairs:
                if i == j:
                    continue
                diff = centers[i] - centers[j]
                d = np.linalg.norm(diff)
                if d > 1e-10 and d < target_dist:
                    # Force proportional to overlap
                    # We want to increase distance d to target_dist
                    # Gradient of (target_dist - d) w.r.t positions is -diff/d
                    # So we add +diff/d to i and -diff/d to j
                    f_mag = (target_dist - d) / d
                    force[i] += f_mag * diff
                    force[j] -= f_mag * diff
            
            # Normalize force to prevent huge jumps
            force_norm = np.linalg.norm(force)
            if force_norm < 1e-10:
                break # No forces, converged
            force = force / force_norm
            
            # Adaptive step size
            step = base_step * (decay_rate ** (it / 10.0))
            
            # Propose new position
            new_centers = centers + step * force
            new_centers = np.clip(new_centers, -0.4999, 0.4999)
            
            new_r = compute_radius(new_centers)
            
            if new_r > r:
                r = new_r
                centers = new_centers
                if r > best_r:
                    best_r = r
                    best_centers = centers.copy()
            else:
                # If step was bad, try smaller step or perturb
                # This helps escape local minima in the force landscape
                step *= 0.5
                new_centers = centers + step * force
                new_centers = np.clip(new_centers, -0.4999, 0.4999)
                new_r = compute_radius(new_centers)
                if new_r > r:
                    r = new_r
                    centers = new_centers
                    if r > best_r:
                        best_r = r
                        best_centers = centers.copy()
                else:
                    # Add small random jitter to escape
                    centers += np.random.normal(0, 0.005, centers.shape)
                    centers = np.clip(centers, -0.4999, 0.4999)
                    r = compute_radius(centers)
                    if r > best_r:
                        best_r = r
                        best_centers = centers.copy()
        
        return best_centers, best_r

    def anneal_and_refine(centers, time_limit, initial_temp=0.05, cooling=0.999):
        """
        Hybrid Simulated Annealing and Force-Directed Refinement.
        Uses SA to jump between basins and force-directed method to settle.
        """
        centers = centers.copy()
        current_r = compute_radius(centers)
        best_centers = centers.copy()
        best_r = current_r
        temp = initial_temp
        
        while time.time() < time_limit and temp > 1e-5:
            # 1. Random Perturbation (SA Move)
            # Move a small number of spheres randomly
            n_move = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            indices = np.random.choice(N, n_move, replace=False)
            step_size = temp * 0.5
            new_centers = centers.copy()
            new_centers[indices] += np.random.normal(0, step_size, (n_move, 3))
            new_centers = np.clip(new_centers, -0.4999, 0.4999)
            
            new_r = compute_radius(new_centers)
            
            # Metropolis Criterion
            delta = new_r - current_r
            if delta > 0 or np.random.random() < np.exp(delta / (temp + 1e-10)):
                current_r = new_r
                centers = new_centers
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            
            temp *= cooling
            
            # 2. Local Refinement (Force-Directed)
            # Occasionally run a short force-directed refinement to settle into a local minimum
            # This accelerates convergence
            if np.random.random() < 0.05:
                refine_limit = min(time.time() + 0.5, time_limit)
                refined_centers, refined_r = force_directed_refine(centers, refine_limit, step_scale=0.5)
                if refined_r > current_r:
                    current_r = refined_r
                    centers = refined_centers
                    if current_r > best_r:
                        best_r = current_r
                        best_centers = centers.copy()
        
        return best_centers, best_r

    # --- Main Execution Loop ---
    
    np.random.seed(42)
    
    best_centers_global = None
    best_radius_global = 0.0
    
    # 1. Generate Initial Configurations
    initializations = []
    
    # Config 1: 5x5x4 Grid (Standard dense packing start)
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.3, 0.3, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    grid_centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Add jittered versions of grid
    for jitter in [0.0, 0.01, 0.02, 0.05]:
        centers = grid_centers + np.random.normal(0, jitter, grid_centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        initializations.append(centers)
        
    # Config 2: FCC-like Lattice
    # FCC unit cell points
    a = 0.35
    fcc_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                fcc_points.append([i*a, j*a, k*a])
                fcc_points.append([i*a + a/2, j*a + a/2, k*a])
                fcc_points.append([i*a + a/2, j*a, k*a + a/2])
                fcc_points.append([i*a, j*a + a/2, k*a + a/2])
    
    fcc_array = np.array(fcc_points)
    mask = np.all(np.abs(fcc_array) <= 0.49, axis=1)
    fcc_array = fcc_array[mask]
    
    if len(fcc_array) >= N:
        idx = np.random.choice(len(fcc_array), N, replace=False)
        fcc_centers = fcc_array[idx]
        for jitter in [0.0, 0.02]:
            centers = fcc_centers + np.random.normal(0, jitter, fcc_centers.shape)
            centers = np.clip(centers, -0.49, 0.49)
            initializations.append(centers)
            
    # Config 3: Random Uniform
    for _ in range(4):
        centers = np.random.uniform(-0.45, 0.45, (N, 3))
        initializations.append(centers)

    # 2. Optimization Loop
    # We will iterate through initializations, spending a time budget on each.
    # If we find a very good solution, we might spend more time refining it.
    
    time_per_init = 15.0 # Seconds per initialization attempt
    
    for idx, init_centers in enumerate(initializations):
        if time.time() > deadline - 5:
            break
        
        remaining = deadline - 5 - time.time()
        if remaining < 2:
            break
            
        # Dynamic time allocation: spend more time on promising starts
        r_init = compute_radius(init_centers)
        if r_init > 0.10:
            t_alloc = min(remaining * 0.5, 20.0)
        elif r_init > 0.08:
            t_alloc = min(remaining * 0.3, 15.0)
        else:
            t_alloc = min(remaining * 0.2, 10.0)
            
        t_limit = time.time() + t_alloc
        
        # Phase A: Annealing to explore
        # We use a shorter annealing phase to find a good basin, then refine
        anneal_time = t_alloc * 0.4
        anneal_limit = time.time() + anneal_time
        
        sa_centers, sa_r = anneal_and_refine(init_centers, anneal_limit, initial_temp=0.08, cooling=0.998)
        
        if sa_r > best_radius_global:
            best_radius_global = sa_r
            best_centers_global = sa_centers
            
        # Phase B: Intensive Force-Directed Refinement on the best found so far
        # This is the key improvement from the inspiration
        refine_time = t_alloc * 0.6
        refine_limit = time.time() + refine_time
        
        # Refine the best center from SA
        fd_centers, fd_r = force_directed_refine(sa_centers, refine_limit, step_scale=1.0)
        
        if fd_r > best_radius_global:
            best_radius_global = fd_r
            best_centers_global = fd_centers
            
        # If we have time, do another round of refinement on the global best
        # to ensure we are in the deepest local minimum
        if time.time() < deadline - 10:
             extra_limit = time.time() + min(5.0, deadline - 10 - time.time())
             fd_centers2, fd_r2 = force_directed_refine(best_centers_global, extra_limit, step_scale=0.5)
             if fd_r2 > best_radius_global:
                 best_radius_global = fd_r2
                 best_centers_global = fd_centers2

    # Final Safety Check
    if best_centers_global is None:
        return grid_centers
        
    # Ensure validity
    best_centers_global = np.clip(best_centers_global, -0.5, 0.5)
    
    return best_centers_global

if __name__ == "__main__":
    # Quick test
    start = time.time()
    result = construct()
    end = time.time()
    r = compute_radius(result)
    print(f"Radius: {r}, Time: {end-start}s")
