"""
Pack 50 equal circles in a unit square using force-directed dynamics 
and coordinate descent polishing.
"""
import numpy as np
import time
import math

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _radius(centers):
    """Compute the max common radius for given centers (exact, for scoring)."""
    # Wall radius = min of (0.5 - |coord|)
    wall_r = np.min(0.5 - np.abs(centers))
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(wall_r, min_pair / 2.0)

def _force_dynamics(centers, target_r, max_time=5.0, dt=0.01):
    """
    Push circles apart to satisfy target_r.
    Returns centers and success flag.
    """
    centers = centers.copy()
    start_time = time.time()
    
    # We iterate until time runs out or forces are negligible.
    # We use a simple velocity-less Verlet-like update: 
    # x_new = x_old + F * dt * damping
    
    while time.time() - start_time < max_time:
        forces = np.zeros_like(centers)
        
        # 1. Wall forces: Push away from walls if closer than target_r
        # If center is too close to a wall, push it in.
        # x < -0.5 + r -> push right (+x)
        # x > 0.5 - r -> push left (-x)
        
        # Calculate violations
        # dist to left wall: x - (-0.5) = x + 0.5
        # if x + 0.5 < r, push right by (r - (x + 0.5))
        
        # Vectorized wall force calculation
        # Left wall: x < -0.5 + r
        viol_l = (target_r - (centers[:, 0] + 0.5))
        forces[:, 0] += np.where(viol_l > 0, viol_l, 0)
        
        # Right wall: x > 0.5 - r
        viol_r = (centers[:, 0] - (0.5 - target_r))
        forces[:, 0] -= np.where(viol_r > 0, viol_r, 0)
        
        # Bottom wall: y < -0.5 + r
        viol_b = (target_r - (centers[:, 1] + 0.5))
        forces[:, 1] += np.where(viol_b > 0, viol_b, 0)
        
        # Top wall: y > 0.5 - r
        viol_t = (centers[:, 1] - (0.5 - target_r))
        forces[:, 1] -= np.where(viol_t > 0, viol_t, 0)
        
        # 2. Pairwise forces: Push apart if closer than 2*target_r
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Overlap amount
        overlap = (2.0 * target_r - dist)
        # Only apply force if overlapping
        overlap = np.where(overlap > 0, overlap, 0)
        
        # Direction unit vector
        # Avoid division by zero
        dist_safe = np.where(dist > 1e-10, dist, 1.0)
        unit_vec = diff / dist_safe[:, :, np.newaxis]
        
        # Force magnitude is proportional to overlap
        # Sum forces from all other particles
        # F_i = sum_j (overlap_ij * unit_vec_ij)
        # Note: unit_vec_ij points from j to i? 
        # diff = i - j. So unit_vec points from j to i.
        # If i is too close to j, we want to push i away from j.
        # So force on i is in direction of (i - j). Correct.
        
        # Mask diagonal to avoid self-interaction (already inf, so overlap 0)
        # Actually, dist is inf on diagonal, so overlap is 0.
        
        pair_forces = np.sum(overlap[:, :, np.newaxis] * unit_vec, axis=1)
        forces += pair_forces
        
        # Update positions
        # Use a damping factor to ensure stability
        damping = 0.5
        centers += forces * dt * damping
        
        # Clip to square to prevent escaping (safety)
        centers[:, 0] = np.clip(centers[:, 0], -0.4999, 0.4999)
        centers[:, 1] = np.clip(centers[:, 1], -0.4999, 0.4999)
        
        # Check if converged (max force small)
        if np.max(np.abs(forces)) < 1e-6:
            break
            
    return centers

def _coordinate_descent_polish(centers, max_time=5.0):
    """
    Polish the configuration by moving points to maximize the minimum distance.
    We iterate over points and try to move them in 8 directions to improve the local min dist.
    """
    centers = centers.copy()
    start_time = time.time()
    step = 0.001
    
    while time.time() - start_time < max_time:
        improved = False
        for i in range(N):
            # Calculate current bottleneck for point i
            # 1. Wall dist
            wall_dists = [
                centers[i, 0] + 0.5,
                0.5 - centers[i, 0],
                centers[i, 1] + 0.5,
                0.5 - centers[i, 1]
            ]
            min_wall = min(wall_dists)
            
            # 2. Pairwise dists
            dists = np.linalg.norm(centers - centers[i], axis=1)
            dists[i] = np.inf
            min_pair = np.min(dists)
            
            current_min = min(min_wall, min_pair)
            
            # Try moving in 8 directions
            dirs = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (-1, 1), (1, -1), (-1, -1)
            ]
            
            best_dir = None
            best_new_min = current_min
            
            for dx, dy in dirs:
                new_c = centers[i].copy()
                new_c[0] += dx * step
                new_c[1] += dy * step
                
                # Check bounds
                if new_c[0] < -0.5 or new_c[0] > 0.5 or new_c[1] < -0.5 or new_c[1] > 0.5:
                    continue
                
                # Calculate new min for this point
                new_wall_dists = [
                    new_c[0] + 0.5,
                    0.5 - new_c[0],
                    new_c[1] + 0.5,
                    0.5 - new_c[1]
                ]
                new_min_wall = min(new_wall_dists)
                
                new_dists = np.linalg.norm(centers - new_c, axis=1)
                new_dists[i] = np.inf
                new_min_pair = np.min(new_dists)
                
                new_min = min(new_min_wall, new_min_pair)
                
                if new_min > best_new_min + 1e-9:
                    best_new_min = new_min
                    best_dir = (dx, dy)
            
            if best_dir is not None:
                centers[i][0] += best_dir[0] * step
                centers[i][1] += best_dir[1] * step
                improved = True
        
        if not improved:
            # No improvement found, reduce step size
            step *= 0.5
            if step < 1e-5:
                break

    return centers

def _make_grid_rect(n_x, n_y):
    """Create a rectangular grid."""
    ticks_x = (np.arange(n_x) + 0.5) / n_x - 0.5
    ticks_y = (np.arange(n_y) + 0.5) / n_y - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    return grid[:N].copy()

def _make_hexagonal_grid(n_x, n_y, spacing, offset_x=0, offset_y=0):
    """Create a hexagonal lattice of points."""
    points = []
    dy = spacing * np.sqrt(3) / 2
    for j in range(n_y):
        for i in range(n_x):
            x = i * spacing + offset_x
            y = j * dy + offset_y
            if j % 2 != 0:
                x += spacing / 2
            points.append((x, y))
    points = np.array(points)
    return points[:N]

def construct():
    global DEADLINE
    start_time = time.time()
    
    best_centers = None
    best_r = 0.0
    
    # Generate initial configurations
    inits = []
    
    # 1. Rectangular Grid 7x8
    inits.append(_make_grid_rect(7, 8))
    
    # 2. Rectangular Grid 8x7
    inits.append(_make_grid_rect(8, 7))
    
    # 3. Hexagonal Grids
    for sp in [0.14, 0.15, 0.16]:
        c_hex = _make_hexagonal_grid(10, 10, sp, offset_x=-0.5, offset_y=-0.5)
        c_hex = c_hex - np.mean(c_hex, axis=0)
        max_ext = np.max(np.abs(c_hex))
        if max_ext > 0.45:
            c_hex = c_hex * (0.45 / max_ext)
        inits.append(c_hex)
        
    # 4. Random
    np.random.seed(42)
    inits.append(np.random.uniform(-0.45, 0.45, size=(N, 2)))
    
    # Strategy:
    # For each init:
    # 1. Estimate current radius.
    # 2. Use Force Dynamics to expand to a target radius slightly larger than current.
    #    We will try to push towards a "global" target radius, e.g., 0.07.
    #    Actually, better to expand in steps.
    #    Let's try to expand to a fixed target r_target = 0.071 (near known best).
    #    If it fails, we fall back to current.
    
    r_target_global = 0.0714 # Slightly above known best
    
    for i, init in enumerate(inits):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        time_left = DEADLINE - 10 - (time.time() - start_time)
        
        # Phase 1: Force Dynamics Expansion
        # We want to find a feasible packing with large r.
        # Let's try to push the init towards r_target_global.
        # If time allows, we can try multiple targets or just one strong push.
        
        # Allocate time for dynamics
        alloc_dyn = min(10.0, time_left * 0.6)
        
        if alloc_dyn > 1.0:
            expanded = _force_dynamics(init, r_target_global, max_time=alloc_dyn)
            
            # Check if it actually achieved the target or close
            r_exp = _radius(expanded)
            
            # If it failed to reach target, maybe it got stuck.
            # We can try a smaller target if time permits, but let's just polish what we have.
            
            if r_exp > best_r:
                best_r = r_exp
                best_centers = expanded.copy()
            
            # Phase 2: Coordinate Descent Polish
            if time.time() - start_time < DEADLINE - 5:
                alloc_polish = min(10.0, (DEADLINE - 5) - (time.time() - start_time))
                if alloc_polish > 1.0:
                    polished = _coordinate_descent_polish(expanded, max_time=alloc_polish)
                    r_pol = _radius(polished)
                    
                    if r_pol > best_r:
                        best_r = r_pol
                        best_centers = polished.copy()
        
    # If time remains, do more random restarts with just polish (faster)
    # Or short dynamics.
    while time.time() - start_time < DEADLINE - 5:
        seed = int(time.time() * 1000) % 100000
        np.random.seed(seed)
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        
        time_left = DEADLINE - 5 - (time.time() - start_time)
        if time_left < 1: break
        
        # Short dynamics
        alloc_dyn = min(5.0, time_left * 0.5)
        if alloc_dyn > 0.5:
            expanded = _force_dynamics(c_rand, r_target_global, max_time=alloc_dyn)
            
            # Polish
            alloc_polish = min(5.0, (DEADLINE - 5) - (time.time() - start_time))
            if alloc_polish > 0.5:
                polished = _coordinate_descent_polish(expanded, max_time=alloc_polish)
                r_pol = _radius(polished)
                
                if r_pol > best_r:
                    best_r = r_pol
                    best_centers = polished.copy()

    return best_centers
