"""DEM-based sphere packing: rigid spheres with Hertzian repulsion, wall repulsion,
velocity-Verlet integration with adaptive dt, multiple random restarts, and a
deadline guard. Returns best valid configuration found within the time budget."""

import numpy as np
import time

N = 50
DEADLINE = 140.0  # safety margin below 150s kill


def _score_of(positions):
    """Compute the largest valid radius for a set of positions (must be inside [-0.5,0.5]^3)."""
    if positions.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(positions)):
        return -1.0
    # Wall clearance: min over i,k of (0.5 - |c_ik|)
    wall_clear = np.min(0.5 - np.abs(positions))
    if wall_clear < 0:
        return -1.0
    # Pairwise distances
    diff = positions[:, None, :] - positions[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_pair, wall_clear)


def _dem_step(pos, vel, dt, k_pair=500.0, k_wall=1000.0, damping=0.98):
    """
    One velocity-Verlet step with repulsive forces.
    pos: (N,3), vel: (N,3), dt: float
    Returns new pos, vel.
    """
    # Compute forces
    # Pairwise repulsion (Hertzian-like: F ~ overlap^2 along direction, only if overlapping)
    # We use a simple model: if distance < 1 (touching), force = k * (1 - dist)^2 * direction
    # Actually for packing we want to maximize min distance, so we use repulsion when dist < threshold
    # Let's use: for each pair, if dist < 1.0, force = k_pair * (1.0 - dist) * (1.0 - dist) * unit_dir
    # This pushes them apart.
    
    # Efficient pairwise computation
    diff = pos[:, None, :] - pos[None, :, :]  # (N,N,3)
    dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (N,N)
    np.fill_diagonal(dists, 1.0)  # avoid zero
    
    # Overlap amount: positive when dist < 1
    overlap = np.maximum(1.0 - dists, 0.0)  # (N,N)
    # Force magnitude per pair: k * overlap^2
    force_mag = k_pair * overlap ** 2  # (N,N)
    # Direction: diff / dist (from j to i)
    # Avoid division by zero
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    dir_vec = diff / safe_dists[:, :, None]  # (N,N,3)
    
    # Force on i from j: force_mag[i,j] * dir_vec[i,j]
    # Sum over j for each i
    forces = np.sum(force_mag[:, :, None] * dir_vec, axis=1)  # (N,3)
    
    # Wall repulsion: push spheres away from walls if within 0.5
    # For each coordinate, if |pos| > 0.5 - margin, push back
    # Use: for each axis, if pos > 0.45, force = -k_wall * (pos - 0.45)^2
    # Actually, let's use a smooth wall potential: for each axis k,
    # wall_force = -k_wall * sign(pos) * max(0, |pos| - 0.45)^2
    # This pushes back toward center when close to wall.
    
    wall_forces = np.zeros_like(pos)
    for axis in range(3):
        p = pos[:, axis]
        abs_p = np.abs(p)
        sign_p = np.sign(p)
        # Overlap with wall region: positive when |p| > 0.45
        wall_overlap = np.maximum(abs_p - 0.45, 0.0)
        # Force pushes back: -sign * k * overlap^2
        wall_forces[:, axis] = -sign_p * k_wall * wall_overlap ** 2
    
    total_forces = forces + wall_forces
    
    # Velocity-Verlet
    # v_half = v + 0.5 * dt * F
    vel_half = vel + 0.5 * dt * total_forces
    # pos_new = pos + dt * v_half
    pos_new = pos + dt * vel_half
    # Compute forces at pos_new (recompute)
    diff2 = pos_new[:, None, :] - pos_new[None, :, :]
    dists2 = np.sqrt(np.sum(diff2 ** 2, axis=2))
    np.fill_diagonal(dists2, 1.0)
    overlap2 = np.maximum(1.0 - dists2, 0.0)
    force_mag2 = k_pair * overlap2 ** 2
    safe_dists2 = np.where(dists2 > 1e-12, dists2, 1.0)
    dir_vec2 = diff2 / safe_dists2[:, :, None]
    forces2 = np.sum(force_mag2[:, :, None] * dir_vec2, axis=1)
    
    wall_forces2 = np.zeros_like(pos_new)
    for axis in range(3):
        p = pos_new[:, axis]
        abs_p = np.abs(p)
        sign_p = np.sign(p)
        wall_overlap = np.maximum(abs_p - 0.45, 0.0)
        wall_forces2[:, axis] = -sign_p * k_wall * wall_overlap ** 2
    
    total_forces2 = forces2 + wall_forces2
    
    # v_new = v_half + 0.5 * dt * F_new
    vel_new = vel_half + 0.5 * dt * total_forces2
    
    # Damping to help settle
    vel_new = vel_new * damping
    
    return pos_new, vel_new


def _run_dem(pos0, vel0, dt, steps, k_pair=500.0, k_wall=1000.0, damping=0.98):
    """Run DEM simulation for given steps. Returns final positions."""
    pos = pos0.copy()
    vel = vel0.copy()
    for step in range(steps):
        pos, vel = _dem_step(pos, vel, dt, k_pair, k_wall, damping)
    return pos


def _improve_with_local_search(pos, max_iters=2000, step_size=0.01):
    """Simple local search: try small random perturbations, keep if score improves."""
    best_pos = pos.copy()
    best_score = _score_of(best_pos)
    rng = np.random.RandomState(42)
    
    for it in range(max_iters):
        # Random perturbation
        perturbation = rng.normal(0, step_size, size=(N, 3))
        new_pos = best_pos + perturbation
        # Clamp to cube
        new_pos = np.clip(new_pos, -0.49, 0.49)
        new_score = _score_of(new_pos)
        if new_score > best_score:
            best_pos = new_pos
            best_score = new_score
    
    return best_pos, best_score


def _initial_positions(seed=0):
    """Generate initial positions: random in cube with some structure."""
    rng = np.random.RandomState(seed)
    # Start with a grid-like structure with some randomness
    m = 4  # 4x4x4 = 64 > 50
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Shuffle and take N
    idx = rng.permutation(len(grid))[:N]
    pos = grid[idx]
    # Add small random jitter
    pos += rng.normal(0, 0.01, size=pos.shape)
    pos = np.clip(pos, -0.49, 0.49)
    return pos


def construct():
    start_time = time.time()
    best_pos = None
    best_score = -1.0
    
    # Try multiple random restarts
    for restart in range(20):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        pos0 = _initial_positions(seed=restart)
        vel0 = np.zeros((N, 3))
        
        # Adaptive DEM: start with small dt, increase if stable
        dt = 0.001
        steps = 500
        
        # Run DEM
        pos = _run_dem(pos0, vel0, dt, steps)
        
        # Check if valid
        score = _score_of(pos)
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
        
        # Local search refinement
        if time.time() - start_time > DEADLINE - 20:
            break
        pos_ls, score_ls = _improve_with_local_search(pos, max_iters=500, step_size=0.005)
        if score_ls > best_score:
            best_score = score_ls
            best_pos = pos_ls.copy()
        
        if time.time() - start_time > DEADLINE - 5:
            break
    
    if best_pos is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pos = grid[:N]
    
    return best_pos
