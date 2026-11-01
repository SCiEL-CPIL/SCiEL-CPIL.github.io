"""Optimized sphere packing in a unit cube via lattice-gas Metropolis annealing."""
import numpy as np
import time

N = 100
HALF = 0.5  # half side of cube


def _score(pos):
    """Compute the minimum clearance r for positions pos of shape (N,3)."""
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Wall clearance
    wall = HALF - np.abs(pos)
    min_wall = np.min(wall)
    return min(min_pair, min_wall)


def _compute_forces(pos, r_target):
    """
    Compute repulsive forces to push spheres apart when closer than 2*r_target.
    Also pushes spheres away from walls when closer than r_target.
    Returns force array of shape (N, 3).
    """
    n = pos.shape[0]
    forces = np.zeros_like(pos)
    
    # Pairwise repulsion
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Only apply force when distance < 2*r_target
    mask = dists < 2 * r_target
    for i in range(n):
        for j in range(i + 1, n):
            if mask[i, j]:
                d = dists[i, j]
                if d < 1e-12:
                    # Random small nudge
                    forces[i] += np.random.randn(3) * 0.01
                    forces[j] -= np.random.randn(3) * 0.01
                else:
                    strength = (2 * r_target - d) / (2 * r_target)
                    direction = diff[i, j] / d
                    forces[i] += strength * direction
                    forces[j] -= strength * direction
    
    # Wall repulsion
    for i in range(n):
        for k in range(3):
            if pos[i, k] > HALF - r_target:
                forces[i, k] -= (pos[i, k] - (HALF - r_target)) / r_target
            elif pos[i, k] < -HALF + r_target:
                forces[i, k] += ((-HALF + r_target) - pos[i, k]) / r_target
    
    return forces


def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # safety margin
    
    # Initial position: simple grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pos = grid[:N].copy()
    
    best_pos = pos.copy()
    best_r = _score(best_pos)
    
    # Annealing parameters
    T = 0.05
    T_min = 0.001
    cooling = 0.999
    step_size = 0.02
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Compute current score
        current_r = _score(pos)
        if current_r > best_r:
            best_r = current_r
            best_pos = pos.copy()
        
        # Propose move: pick random sphere and perturb
        idx = rng.integers(0, N)
        displacement = rng.normal(0, step_size, 3)
        new_pos = pos.copy()
        new_pos[idx] += displacement
        
        # Keep within bounds
        new_pos[idx] = np.clip(new_pos[idx], -HALF + 1e-9, HALF - 1e-9)
        
        new_r = _score(new_pos)
        
        # Metropolis acceptance
        delta_r = new_r - current_r
        if delta_r > 0:
            pos = new_pos
        else:
            if T > T_min:
                accept_prob = np.exp(delta_r / T)
                if rng.random() < accept_prob:
                    pos = new_pos
        
        # Cool down
        T = max(T * cooling, T_min)
        
        # Occasionally reset temperature if stuck
        if T == T_min and rng.random() < 0.01:
            T = 0.02
    
    return best_pos
