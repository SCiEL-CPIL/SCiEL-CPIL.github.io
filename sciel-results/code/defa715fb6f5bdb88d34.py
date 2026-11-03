"""Improved sphere packing: start from a perturbed grid, then local optimization."""
import numpy as np
import time

N = 100

def _score(c):
    """Compute the packing radius for centers c of shape (N, 3)."""
    # Distance to walls
    wall = 0.5 - np.max(np.abs(c), axis=1)
    min_wall = np.min(wall)
    # Pairwise distances
    # Use broadcasting
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # Set diagonal to inf
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    return min(min_wall, min_pair)

def _local_optimize(c, lr=0.01, max_iter=500, eps=1e-7):
    """Simple gradient-free local optimization: move each sphere to maximize min constraint."""
    best_c = c.copy()
    best_r = _score(best_c)
    
    for it in range(max_iter):
        moved = False
        r = _score(best_c)
        
        # Try to improve the worst constraint
        wall = 0.5 - np.max(np.abs(best_c), axis=1)
        diff = best_c[:, np.newaxis, :] - best_c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        pair_dist = dist / 2.0
        
        # Find the most violated constraint
        min_wall_idx = np.argmin(wall)
        min_wall_val = wall[min_wall_idx]
        
        i_idx, j_idx = np.unravel_index(np.argmin(pair_dist), pair_dist.shape)
        min_pair_val = pair_dist[i_idx, j_idx]
        
        if min_wall_val < min_pair_val:
            # Move sphere min_wall_idx away from nearest wall
            ci = best_c[min_wall_idx]
            axis = np.argmax(np.abs(ci))
            sign = np.sign(ci[axis])
            if sign == 0:
                sign = 1
            step = min(lr, 0.005)
            new_ci = ci.copy()
            new_ci[axis] -= sign * step
            # Keep within bounds
            new_ci = np.clip(new_ci, -0.499, 0.499)
            test_c = best_c.copy()
            test_c[min_wall_idx] = new_ci
            new_r = _score(test_c)
            if new_r > r + eps:
                best_c = test_c
                best_r = new_r
                moved = True
        else:
            # Move spheres i and j apart
            ci = best_c[i_idx]
            cj = best_c[j_idx]
            d = ci - cj
            norm = np.linalg.norm(d)
            if norm < 1e-10:
                d = np.array([1.0, 0, 0])
                norm = 1.0
            d = d / norm
            step = min(lr, 0.005)
            new_ci = ci + d * step
            new_cj = cj - d * step
            # Keep within bounds
            new_ci = np.clip(new_ci, -0.499, 0.499)
            new_cj = np.clip(new_cj, -0.499, 0.499)
            test_c = best_c.copy()
            test_c[i_idx] = new_ci
            test_c[j_idx] = new_cj
            new_r = _score(test_c)
            if new_r > r + eps:
                best_c = test_c
                best_r = new_r
                moved = True
        
        if not moved:
            # Try random perturbations
            improved = False
            for _ in range(20):
                test_c = best_c.copy()
                idx = np.random.randint(N)
                test_c[idx] += np.random.randn(3) * 0.01
                test_c = np.clip(test_c, -0.499, 0.499)
                new_r = _score(test_c)
                if new_r > best_r + eps:
                    best_c = test_c
                    best_r = new_r
                    improved = True
                    break
            if not improved:
                break
    
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # Initial grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    c = grid[:N].copy()
    best_c = c.copy()
    best_r = _score(best_c)
    
    # Local optimization
    c, r = _local_optimize(c, lr=0.01, max_iter=1000)
    if r > best_r:
        best_c = c
        best_r = r
    
    elapsed = time.time() - start_time
    if elapsed < deadline:
        # Try more optimization with smaller step
        c, r = _local_optimize(best_c, lr=0.003, max_iter=1000)
        if r > best_r:
            best_c = c
            best_r = r
    
    elapsed = time.time() - start_time
    if elapsed < deadline:
        # Random restarts
        rng = np.random.default_rng(42)
        while time.time() - start_time < deadline:
            # Perturb from best
            c = best_c + rng.normal(0, 0.02, (N, 3))
            c = np.clip(c, -0.499, 0.499)
            c, r = _local_optimize(c, lr=0.01, max_iter=300)
            if r > best_r:
                best_c = c
                best_r = r
    
    return best_c
