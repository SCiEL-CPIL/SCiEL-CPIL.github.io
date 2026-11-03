"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(c):
    """Compute the common radius for centres c (N,3)."""
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Distance between all pairs
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Distance to walls
    wall = 0.5 - np.max(np.abs(c), axis=1)
    min_wall = np.min(wall)
    return min(min_pair, min_wall)

def _repulsive_force(c, r_target):
    """
    Compute repulsive forces to push spheres apart and keep them in the cube.
    Uses a soft potential: for each pair, if dist < 2*r_target, push apart.
    Also push away from walls if too close.
    """
    forces = np.zeros_like(c)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, 1e-10)
    
    # Pairwise repulsion
    # Only consider pairs that are too close
    mask = dists < 2.0 * r_target
    for i in range(N):
        for j in range(i + 1, N):
            if mask[i, j]:
                d = dists[i, j]
                if d < 1e-10:
                    # Random nudge
                    forces[i] += np.random.randn(3) * 0.01
                    forces[j] -= np.random.randn(3) * 0.01
                else:
                    # Repulsive force proportional to overlap
                    overlap = 2.0 * r_target - d
                    f = overlap / d
                    forces[i] += f * diff[i, j]
                    forces[j] -= f * diff[i, j]
    
    # Wall repulsion
    for i in range(N):
        for k in range(3):
            # Left wall
            if c[i, k] < -0.5 + r_target:
                forces[i, k] += (-0.5 + r_target - c[i, k]) * 10.0
            # Right wall
            if c[i, k] > 0.5 - r_target:
                forces[i, k] -= (c[i, k] - (0.5 - r_target)) * 10.0
    
    return forces

def _local_optimize(c, r_target, max_iter=1000, tol=1e-8):
    """
    Local optimization using gradient ascent on a potential that encourages
    minimum distance to be at least 2*r_target.
    """
    c = c.copy()
    lr = 0.01
    for it in range(max_iter):
        f = _repulsive_force(c, r_target)
        c_new = c + lr * f
        # Project back into cube
        c_new = np.clip(c_new, -0.5 + r_target * 0.5, 0.5 - r_target * 0.5)
        
        # Check convergence
        if np.max(np.abs(c_new - c)) < tol:
            c = c_new
            break
        c = c_new
    
    return c

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Start with a reasonable initial configuration
    # Use a body-centered cubic arrangement as starting point
    # For 50 spheres, we can use a 4x4x4 grid with some adjustments
    
    # Initial: spread out on a grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c = grid[:N].copy()
    
    best_c = c.copy()
    best_r = _score(c)
    
    # Try different starting configurations
    # 1. Simple grid
    # 2. Random with jitter
    # 3. FCC-like arrangement
    
    for restart in range(10):
        if time.time() > deadline:
            break
        
        if restart == 0:
            # Already have grid
            pass
        elif restart == 1:
            # Random
            c = np.random.uniform(-0.4, 0.4, (N, 3))
        elif restart == 2:
            # FCC-like: alternate layers
            c = np.zeros((N, 3))
            idx = 0
            for layer in range(5):
                z = (layer - 2) * 0.2
                for row in range(4):
                    y = (row - 1.5) * 0.25
                    for col in range(3):
                        if idx >= N:
                            break
                        x = (col - 1) * 0.3 + (0.1 if row % 2 else 0)
                        c[idx] = [x, y, z]
                        idx += 1
                    if idx >= N:
                        break
                if idx >= N:
                    break
            c = np.clip(c, -0.45, 0.45)
        else:
            # Perturbed best
            c = best_c + np.random.randn(N, 3) * 0.05
            c = np.clip(c, -0.45, 0.45)
        
        # Local optimization
        r_target = best_r + 0.001
        for opt_iter in range(5):
            if time.time() > deadline:
                break
            c = _local_optimize(c, r_target, max_iter=500, tol=1e-7)
            r_current = _score(c)
            
            if r_current > best_r:
                best_r = r_current
                best_c = c.copy()
                r_target = r_current + 0.0005
            else:
                # Decrease target and try again
                r_target = r_current + 0.0002
        
        # Final polish
        if time.time() < deadline - 5.0:
            r_target = best_r + 0.0001
            for _ in range(3):
                best_c = _local_optimize(best_c, r_target, max_iter=200, tol=1e-8)
                r_current = _score(best_c)
                if r_current > best_r:
                    best_r = r_current
                    r_target = r_current + 0.00005
    
    return best_c
# EVOLVE-BLOCK-END
