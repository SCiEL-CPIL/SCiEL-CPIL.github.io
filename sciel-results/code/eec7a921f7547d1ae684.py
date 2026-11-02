"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _score(c):
    """Compute the radius for a configuration of centres."""
    n = len(c)
    if n < 2:
        return np.inf
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to walls
    coords = np.abs(c)
    min_wall = 0.5 - np.max(coords, axis=1)
    min_wall = np.min(min_wall)
    
    return min(min_pair, min_wall)

def _initial_guess(seed=42):
    """Generate an initial guess using a jittered grid."""
    rng = np.random.RandomState(seed)
    m = int(math.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c = grid[:N].copy()
    # Add small random jitter
    c += rng.uniform(-0.01, 0.01, c.shape)
    # Clip to stay inside
    c = np.clip(c, -0.49, 0.49)
    return c

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_c = _initial_guess(42)
    best_r = _score(best_c)
    
    # Try multiple random restarts with different seeds
    for seed in range(42, 42 + 100):
        if time.time() > deadline:
            break
        
        c = _initial_guess(seed)
        r = _score(c)
        
        # Simple local optimization using gradient-free method (Nelder-Mead style)
        # We'll use a simple iterative improvement scheme
        improved = True
        max_iter = 2000
        step_size = 0.01
        current_r = r
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Find the bottleneck constraint
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
            
            coords = np.abs(c)
            min_wall_dist = 0.5 - np.max(coords, axis=1)
            min_wall_idx = np.argmin(min_wall_dist)
            
            # Determine if pair or wall is bottleneck
            if min_pair_dist / 2.0 <= min_wall_dist[min_wall_idx]:
                # Pair is bottleneck
                i, j = min_pair_idx
                if i == j:
                    continue
                # Push i and j apart
                direction = c[i] - c[j]
                norm = np.linalg.norm(direction)
                if norm < 1e-10:
                    direction = np.random.randn(3)
                    norm = np.linalg.norm(direction)
                direction /= norm
                c[i] += direction * step_size * 0.5
                c[j] -= direction * step_size * 0.5
            else:
                # Wall is bottleneck
                k = min_wall_idx
                # Push away from the closest wall
                for axis in range(3):
                    if np.abs(c[k, axis]) == np.max(np.abs(c[k])):
                        if c[k, axis] > 0:
                            c[k, axis] -= step_size
                        else:
                            c[k, axis] += step_size
                        break
                else:
                    c[k, 0] += step_size * np.sign(c[k, 0]) if c[k, 0] != 0 else step_size
            
            # Clip to stay inside
            c = np.clip(c, -0.499, 0.499)
            
            new_r = _score(c)
            if new_r > current_r + 1e-12:
                current_r = new_r
            else:
                # Revert and try smaller step
                c = np.clip(c, -0.499, 0.499)
                # Try a different perturbation
                c += np.random.randn(3) * 0.001
                c = np.clip(c, -0.499, 0.499)
                new_r = _score(c)
                if new_r > current_r + 1e-12:
                    current_r = new_r
                else:
                    # Accept anyway to escape local minima
                    current_r = new_r
            
            if current_r > best_r:
                best_r = current_r
                best_c = c.copy()
        
        if current_r > best_r:
            best_r = current_r
            best_c = c.copy()
        
        # Anneal step size
        step_size *= 0.95
    
    # Final cleanup: ensure all points are strictly inside
    best_c = np.clip(best_c, -0.4999, 0.4999)
    
    return best_c
# EVOLVE-BLOCK-END
