"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START

def _score(c):
    """Compute the packing radius for centers c (N,2) in [-0.5,0.5]^2."""
    n = len(c)
    if n == 0:
        return 0.0
    # Boundary constraint
    bmin = 0.5 - np.abs(c).max(axis=1)
    r_bound = bmin.min()
    # Pairwise distances
    # For efficiency, compute all pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    r_pair = dists.min() / 2.0
    return min(r_bound, r_pair)


def _init_grid():
    """Initialize with a square grid, slightly perturbed."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    grid = grid[:N].copy()
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    grid += rng.normal(0, 0.001, size=grid.shape)
    # Clamp to square
    grid = np.clip(grid, -0.49, 0.49)
    return grid


def _subgradient_descent(c, max_iters=5000, step_scale=0.001, decay=0.9995):
    """Stochastic subgradient ascent on the packing radius."""
    c = c.copy()
    n = len(c)
    rng = np.random.default_rng(123)
    
    # Precompute index pairs for efficiency
    # We'll use a random subset of pairs each iteration
    
    best_r = _score(c)
    best_c = c.copy()
    
    for it in range(max_iters):
        # Compute current radius and identify active constraints
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        bmin = 0.5 - np.abs(c)
        
        # Find the minimum pairwise distance
        min_pair_dist = dists.min()
        r_pair = min_pair_dist / 2.0
        
        # Find minimum boundary distance
        r_bound = bmin.min()
        
        r = min(r_pair, r_bound)
        
        if r > best_r:
            best_r = r
            best_c = c.copy()
        
        # Build gradient for radius
        grad = np.zeros_like(c)
        
        if r_bound <= r_pair:
            # Boundary constraint is active
            idx = np.argmin(bmin)
            k = np.argmin(bmin[idx])  # which coordinate
            # Gradient: push away from boundary
            if c[idx, k] > 0:
                grad[idx, k] = -1.0
            else:
                grad[idx, k] = 1.0
        else:
            # Pairwise constraint is active
            # Find the pair with minimum distance
            i, j = np.unravel_index(np.argmin(dists), dists.shape)
            if i > j:
                i, j = j, i
            d = c[i] - c[j]
            dist = np.linalg.norm(d)
            if dist > 1e-12:
                dir_vec = d / dist
                # To increase distance, push i in direction of dir_vec and j opposite
                grad[i] += dir_vec / 2.0
                grad[j] -= dir_vec / 2.0
        
        # Apply step
        step_size = step_scale / (1.0 + it * 0.001)
        c += step_size * grad
        
        # Clamp to square
        c = np.clip(c, -0.499, 0.499)
        
        # Periodically recompute best
        if it % 100 == 0:
            r_check = _score(c)
            if r_check > best_r:
                best_r = r_check
                best_c = c.copy()
    
    return best_c, best_r


def _local_optimize(c, max_iters=3000):
    """Refine using a more careful subgradient method with multiple active constraints."""
    c = c.copy()
    n = len(c)
    rng = np.random.default_rng(456)
    
    best_r = _score(c)
    best_c = c.copy()
    
    # Use a list of the k closest pairs and closest boundaries
    k_pairs = 10
    
    for it in range(max_iters):
        # Compute all pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        bmin = 0.5 - np.abs(c)
        
        # Get k smallest pairwise distances
        flat_dists = dists.flatten()
        k_idx = np.argpartition(flat_dists, k_pairs)[:k_pairs]
        
        # Get k smallest boundary distances
        flat_bmin = bmin.flatten()
        k_bidx = np.argpartition(flat_bmin, min(k_pairs, n*2))[:min(k_pairs, n*2)]
        
        # Build gradient from active constraints
        grad = np.zeros_like(c)
        weights = np.zeros_like(flat_dists)
        
        # Weight by inverse distance (closer pairs get more weight)
        for idx in k_idx:
            i, j = np.unravel_index(idx, dists.shape)
            if i > j:
                i, j = j, i
            d = c[i] - c[j]
            dist = np.linalg.norm(d)
            if dist > 1e-12:
                w = 1.0 / (dist + 0.001)
                dir_vec = d / dist
                grad[i] += w * dir_vec / 2.0
                grad[j] -= w * dir_vec / 2.0
        
        for idx in k_bidx:
            ci, ck = np.unravel_index(idx, bmin.shape)
            val = bmin[ci, ck]
            w = 1.0 / (val + 0.001)
            if c[ci, ck] > 0:
                grad[ci, ck] -= w
            else:
                grad[ci, ck] += w
        
        # Normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-12:
            grad /= gnorm
        
        # Adaptive step size
        step_size = 0.0005 / (1.0 + it * 0.0005)
        c += step_size * grad
        c = np.clip(c, -0.499, 0.499)
        
        if it % 50 == 0:
            r_check = _score(c)
            if r_check > best_r:
                best_r = r_check
                best_c = c.copy()
    
    return best_c, best_r


def construct():
    """Construct the circle packing."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with grid initialization
    c = _init_grid()
    best_c = c.copy()
    best_r = _score(c)
    
    # Phase 1: Coarse subgradient descent
    c, r = _subgradient_descent(c, max_iters=2000)
    if r > best_r:
        best_r = r
        best_c = c.copy()
    
    if time.time() > deadline - 60.0:
        return best_c
    
    # Phase 2: Local refinement
    c, r = _local_optimize(best_c, max_iters=2000)
    if r > best_r:
        best_r = r
        best_c = c.copy()
    
    if time.time() > deadline - 30.0:
        return best_c
    
    # Phase 3: Further refinement with smaller steps
    c2 = best_c + np.random.default_rng(789).normal(0, 0.0005, size=best_c.shape)
    c2 = np.clip(c2, -0.499, 0.499)
    c2, r2 = _local_optimize(c2, max_iters=1500)
    if r2 > best_r:
        best_r = r2
        best_c = c2.copy()
    
    return best_c

# EVOLVE-BLOCK-END
