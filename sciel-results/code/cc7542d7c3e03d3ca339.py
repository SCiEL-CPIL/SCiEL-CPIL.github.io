"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _compute_radius(centres):
    """Compute the packing radius for a given set of centres."""
    n = centres.shape[0]
    if n != N:
        return -1.0
    if not np.all(np.isfinite(centres)):
        return -1.0
    if np.any(centres < -0.5 - 1e-10) or np.any(centres > 0.5 + 1e-10):
        return -1.0
    c = np.clip(centres, -0.5, 0.5)
    wall_dists = np.min(0.5 - np.abs(c), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall < -1e-10:
        return -1.0
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    r = min(min_wall, min_pair)
    return r

def _force_directed_relaxation(centres, max_iters=5000, temp_init=0.01, temp_min=1e-6, cool_rate=0.995):
    """
    Force-directed relaxation using Lennard-Jones-like repulsion and wall confinement.
    Uses simulated annealing to escape local minima.
    """
    c = centres.copy()
    current_r = _compute_radius(c)
    if current_r < 0:
        return c
    
    # LJ parameters: sigma ~ typical spacing, epsilon for energy scale
    # We'll use a simplified repulsive force model
    sigma = 0.1  # characteristic length scale
    eps_lj = 1.0
    
    temperature = temp_init
    best_c = c.copy()
    best_r = current_r
    
    for it in range(max_iters):
        if temperature < temp_min:
            break
            
        # Compute forces on each sphere
        forces = np.zeros_like(c)
        
        # Pairwise repulsive forces (LJ-like, truncated)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Only consider pairs within cutoff to save time
        cutoff = 0.3
        mask = dists < cutoff
        mask &= ~np.eye(N, dtype=bool)
        
        # LJ force: F = 24*eps*(2*(sigma/r)^12 - (sigma/r)^6) / r^2 * r_vec
        # Simplified: use inverse-cube repulsion for stability
        for i in range(N):
            for j in np.where(mask[i])[0]:
                r_vec = diff[i, j]
                r = dists[i, j]
                if r < 1e-10:
                    continue
                # Repulsive force magnitude (inverse cube)
                f_mag = 1.0 / (r**3)
                forces[i] += f_mag * r_vec
                forces[j] -= f_mag * r_vec
        
        # Wall confinement forces: push back if too close to walls
        for i in range(N):
            for k in range(3):
                if c[i, k] < -0.5 + 0.05:
                    forces[i, k] += 10.0 * (-0.5 + 0.05 - c[i, k])
                elif c[i, k] > 0.5 - 0.05:
                    forces[i, k] += 10.0 * (0.5 - 0.05 - c[i, k])
        
        # Propose new position: c_new = c + forces * step_size + noise
        step_size = 0.01 * temperature
        noise = np.random.normal(0, step_size * 0.5, size=c.shape)
        c_new = c + forces * step_size + noise
        c_new = np.clip(c_new, -0.5, 0.5)
        
        r_new = _compute_radius(c_new)
        
        if r_new > current_r:
            # Accept improvement
            c = c_new
            current_r = r_new
            if r_new > best_r:
                best_r = r_new
                best_c = c.copy()
        else:
            # Accept with probability exp((r_new - current_r) / temperature)
            delta_r = r_new - current_r
            if delta_r < 0:
                accept_prob = np.exp(delta_r / (temperature * 0.1))
                if np.random.random() < accept_prob:
                    c = c_new
                    current_r = r_new
            # Otherwise reject, keep c
        
        # Cool down
        temperature *= cool_rate
    
    return best_c

def _grid_initialization():
    """Create a grid-based initial configuration."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N]
    return grid

def _random_initialization():
    """Create a random initial configuration."""
    return np.random.uniform(-0.4, 0.4, size=(N, 3))

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    np.random.seed(42)
    
    best_centres = None
    best_r = -1.0
    
    # Strategy 1: Grid initialization + force-directed relaxation
    grid = _grid_initialization()
    grid_relaxed = _force_directed_relaxation(grid, max_iters=3000, temp_init=0.005, temp_min=1e-7, cool_rate=0.998)
    r_grid = _compute_radius(grid_relaxed)
    if r_grid > best_r:
        best_r = r_grid
        best_centres = grid_relaxed.copy()
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_centres
    
    # Strategy 2: Multiple random restarts with force-directed relaxation
    n_restarts = 0
    max_restarts = 10
    
    while time.time() - start_time < deadline - 10:
        n_restarts += 1
        random_c = _random_initialization()
        
        # Vary temperature and iterations for diversity
        temp_init = 0.002 + 0.003 * (n_restarts % 3)
        max_iters = 2000 + 1000 * (n_restarts % 2)
        
        relaxed = _force_directed_relaxation(random_c, max_iters=max_iters, temp_init=temp_init, temp_min=1e-7, cool_rate=0.998)
        r = _compute_radius(relaxed)
        
        if r > best_r:
            best_r = r
            best_centres = relaxed.copy()
        
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
    
    # Strategy 3: Refine best solution with lower temperature for fine-tuning
    if best_centres is not None and time.time() - start_time < deadline - 5:
        refined = _force_directed_relaxation(best_centres, max_iters=5000, temp_init=0.001, temp_min=1e-8, cool_rate=0.999)
        r_refined = _compute_radius(refined)
        if r_refined > best_r:
            best_r = r_refined
            best_centres = refined.copy()
    
    # Final: ensure validity
    if best_centres is not None:
        best_centres = np.clip(best_centres, -0.5, 0.5)
        final_r = _compute_radius(best_centres)
        if final_r < 0:
            # Fallback to grid
            best_centres = _grid_initialization()
    
    return best_centres
# EVOLVE-BLOCK-END
