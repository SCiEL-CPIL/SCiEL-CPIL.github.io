"""
Packing 50 circles in a unit square using Physics-Based Soft-Body Simulation.
Strategy:
1. Initialize with a grid.
2. Run a soft-body simulation where circles repel each other strongly if overlapping
   and weakly attract if far apart (to keep them centered), while being pushed
   away from boundaries.
3. Use Simulated Annealing to accept moves that increase overlap (decrease radius)
   occasionally to escape local minima.
4. Refine the best configuration found using targeted local optimization on the
   most constraining pairs.
"""

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _compute_radius(centers):
    """
    Compute the maximum common radius for given centers in [-0.5, 0.5]^2.
    r = min( min_{i<j} |c_i - c_j| / 2, min_i min_k (0.5 - |c_ik|) )
    """
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check bounds
    if np.any(centers < -0.5 - 1e-9) or np.any(centers > 0.5 + 1e-9):
        return 0.0
        
    centers = np.clip(centers, -0.5, 0.5)
    
    # Boundary distances
    dist_left = centers[:, 0] + 0.5
    dist_right = 0.5 - centers[:, 0]
    dist_bottom = centers[:, 1] + 0.5
    dist_top = 0.5 - centers[:, 1]
    
    min_boundary = np.min(np.min(np.stack([dist_left, dist_right, dist_bottom, dist_top]), axis=0))
    
    # Pairwise distances
    # Use broadcasting for efficiency, but N=50 is small enough
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pairwise = np.min(dists) / 2.0
    
    return min(min_boundary, min_pairwise)

def _simulate(centers, T=0.01, steps=2000, rng=None):
    """
    Run a physics-based simulation to optimize packing.
    Force model:
    - Repulsion between overlapping circles: F_rep = k_rep * (2r - d) if d < 2r
    - Weak attraction to center to prevent drift: F_attr = k_attr * pos
    - Repulsion from walls: F_wall = k_wall * (r - dist_to_wall) if dist_to_wall < r
    """
    if rng is None:
        rng = np.random.default_rng()
        
    pos = centers.copy()
    r_current = _compute_radius(pos)
    
    # Hyperparameters
    k_rep = 10.0
    k_wall = 5.0
    k_center = 0.01
    
    # We want to maximize r. 
    # In simulation, we let particles find a configuration where they are as far apart as possible.
    # Actually, a better approach for packing is to start with a large radius and shrink it?
    # No, standard approach: treat them as hard spheres with a soft penalty for overlap.
    # Objective to minimize: Sum of overlaps^2 + Wall violations^2
    # But we want to maximize the minimum distance.
    
    # Let's use a Metropolis-style simulated annealing on the radius score.
    # Move: perturb one or more circles.
    # Accept if new_radius > old_radius OR with probability exp((new-old)/T).
    
    best_pos = pos.copy()
    best_r = r_current
    current_r = r_current
    
    for step in range(steps):
        # Cooldown
        T_current = T * (0.99 ** (step / 100.0))
        if T_current < 1e-5:
            T_current = 1e-5
            
        # Choose a move
        # 1. Single particle random move
        if rng.random() < 0.5:
            idx = rng.integers(0, N)
            delta = rng.normal(0, 0.05, size=2)
            new_pos = pos.copy()
            new_pos[idx] += delta
        else:
            # 2. Pair swap or move
            i, j = rng.choice(N, size=2, replace=False)
            delta_i = rng.normal(0, 0.05, size=2)
            delta_j = rng.normal(0, 0.05, size=2)
            new_pos = pos.copy()
            new_pos[i] += delta_i
            new_pos[j] += delta_j
            
        # Clip to square
        new_pos = np.clip(new_pos, -0.5, 0.5)
        
        new_r = _compute_radius(new_pos)
        
        # Acceptance criterion
        if new_r > current_r:
            pos = new_pos
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pos = pos.copy()
        else:
            # Probability of accepting worse solution
            # Delta R is negative
            delta_r = new_r - current_r
            # Scale delta_r to be comparable to T
            # T is around 0.01 initially. Radius changes are small.
            # Let's scale: if delta_r is -0.001, and T is 0.01, prob = exp(-0.1) ~ 0.9
            # This seems reasonable.
            prob = np.exp(delta_r / T_current)
            if rng.random() < prob:
                pos = new_pos
                current_r = new_r
            # Else reject, keep pos
        
    return best_pos, best_r

def _local_refine(centers, iterations=100):
    """
    Targeted local optimization: 
    Identify the pair of circles (or circle-wall) that determines the current radius.
    Try to move them apart.
    """
    pos = centers.copy()
    current_r = _compute_radius(pos)
    
    for _ in range(iterations):
        # Find the constraining pair
        # 1. Boundary
        dist_left = pos[:, 0] + 0.5
        dist_right = 0.5 - pos[:, 0]
        dist_bottom = pos[:, 1] + 0.5
        dist_top = 0.5 - pos[:, 1]
        
        boundary_dists = np.min(np.stack([dist_left, dist_right, dist_bottom, dist_top]), axis=0)
        min_boundary_idx = np.argmin(boundary_dists)
        min_boundary_val = boundary_dists[min_boundary_idx]
        
        # 2. Pairwise
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_pair_val = dists[min_pair_idx] / 2.0
        
        # Determine which is the bottleneck
        if min_boundary_val < min_pair_val:
            # Move particle away from wall
            idx = min_boundary_idx
            # Determine which wall
            if dist_left[idx] == min_boundary_val:
                dx, dy = 1, 0
            elif dist_right[idx] == min_boundary_val:
                dx, dy = -1, 0
            elif dist_bottom[idx] == min_boundary_val:
                dx, dy = 0, 1
            else:
                dx, dy = 0, -1
            
            new_pos = pos.copy()
            new_pos[idx] += np.array([dx, dy]) * 0.01
            new_pos = np.clip(new_pos, -0.5, 0.5)
            
            new_r = _compute_radius(new_pos)
            if new_r > current_r:
                pos = new_pos
                current_r = new_r
        else:
            # Move pair apart
            i, j = min_pair_idx
            vec = pos[i] - pos[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                dir_vec = vec / norm
            else:
                dir_vec = np.array([1.0, 0.0])
                
            new_pos = pos.copy()
            new_pos[i] += dir_vec * 0.005
            new_pos[j] -= dir_vec * 0.005
            new_pos = np.clip(new_pos, -0.5, 0.5)
            
            new_r = _compute_radius(new_pos)
            if new_r > current_r:
                pos = new_pos
                current_r = new_r
            else:
                # Try smaller step or random nudge
                break
                
    return pos, current_r

def _construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_radius = -np.inf
    
    # Initial configurations
    # 1. Grid
    m = 7
    n = 8
    points = []
    for i in range(n):
        for j in range(m):
            if len(points) >= N:
                break
            x = (j + 0.5) / m - 0.5
            y = (i + 0.5) / n - 0.5
            points.append([x, y])
    init_grid = np.array(points[:N])
    
    # 2. Perturbed Grid
    init_perturbed = init_grid + rng.normal(0, 0.02, size=(N, 2))
    init_perturbed = np.clip(init_perturbed, -0.49, 0.49)
    
    # 3. Hexagonal approximation
    points_hex = []
    row_height = 0.09
    col_width = 0.1
    offset = col_width / 2
    for row in range(8):
        y = -0.5 + 0.05 + row * row_height
        for col in range(7):
            if len(points_hex) >= N:
                break
            if row % 2 == 0:
                x = -0.5 + 0.05 + col * col_width
            else:
                x = -0.5 + 0.05 + offset + col * col_width
            points_hex.append([x, y])
    init_hex = np.array(points_hex[:N])
    
    initial_configs = [init_grid, init_perturbed, init_hex]
    
    for init in initial_configs:
        if time.time() > deadline:
            break
            
        # Phase 1: Simulated Annealing / Soft Body
        sa_centers, sa_radius = _simulate(init, T=0.02, steps=3000, rng=rng)
        
        # Phase 2: Local Refinement
        ref_centers, ref_radius = _local_refine(sa_centers, iterations=200)
        
        if ref_radius > best_radius:
            best_radius = ref_radius
            best_centers = ref_centers.copy()
            
        # Phase 3: More SA from the refined result with lower T
        if time.time() < deadline - 20:
            sa2_centers, sa2_radius = _simulate(ref_centers, T=0.005, steps=2000, rng=rng)
            ref2_centers, ref2_radius = _local_refine(sa2_centers, iterations=100)
            
            if ref2_radius > best_radius:
                best_radius = ref2_radius
                best_centers = ref2_centers.copy()
                
    # Final fallback
    if best_centers is None:
        best_centers = init_grid
    else:
        best_centers = np.clip(best_centers, -0.5, 0.5)
        
    return best_centers
# EVOLVE-BLOCK-END

def construct():
    return _construct()
