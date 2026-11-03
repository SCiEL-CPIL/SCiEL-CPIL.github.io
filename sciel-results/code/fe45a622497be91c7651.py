"""Circle packing in unit square via force-directed relaxation + local polish."""
import numpy as np
import time

N = 200


def _radius(c):
    """Compute the packing radius for centers c (N,2) in [-0.5,0.5]^2."""
    # Wall distances
    wall = np.minimum(0.5 - np.abs(c), np.min(0.5 - np.abs(c), axis=1))
    # Actually: min over k of (0.5 - |c_ik|) for each i
    wall_min = np.min(0.5 - np.abs(c), axis=1)
    # Pair distances
    # Use broadcasting for N=200 (40000 pairs, manageable)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_min = np.min(dist, axis=1)
    r = np.min(np.minimum(pair_min / 2.0, wall_min))
    return r


def _force_directed(c, iterations=3000, lr_start=0.001, lr_end=0.00001):
    """Force-directed relaxation to maximize minimum distance."""
    c = c.copy()
    for it in range(iterations):
        lr = lr_start * (lr_end / lr_start) ** (it / max(iterations - 1, 1))
        # Repulsion forces: each pair pushes apart
        # F_i = sum_{j != i} (c_i - c_j) / |c_i - c_j|^3  (Coulomb)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)  # avoid div by zero
        dist = np.sqrt(dist_sq)
        # Force magnitude ~ 1/dist^2, direction = diff/dist
        # F_i = sum_j diff_ij / dist_ij^3
        force = np.sum(diff / (dist_sq * dist)[:, :, np.newaxis], axis=1)
        # Attraction to center
        force -= 0.01 * c
        # Update
        c += lr * force
        # Clamp to square
        c = np.clip(c, -0.499, 0.499)
    return c


def _local_polish(c, max_iters=200, tol=1e-10):
    """Local search on the exact non-smooth objective using Nelder-Mead style."""
    c = c.copy()
    best_r = _radius(c)
    for it in range(max_iters):
        # Find the bottleneck constraint
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        pair_min = np.min(dist, axis=1)
        wall_min = np.min(0.5 - np.abs(c), axis=1)
        current_r = np.min(np.minimum(pair_min / 2.0, wall_min))
        if current_r > best_r:
            best_r = current_r
            c = c.copy()
        # Perturb the most constrained point
        # Find which constraint is tightest
        tight_idx = np.argmin(np.minimum(pair_min / 2.0, wall_min))
        # Try small random perturbations
        improved = False
        for attempt in range(20):
            delta = np.random.randn(N, 2) * 0.001
            delta[tight_idx] *= 2.0  # focus on tight point
            c_new = c + delta
            c_new = np.clip(c_new, -0.499, 0.499)
            r_new = _radius(c_new)
            if r_new > current_r + tol:
                c = c_new
                current_r = r_new
                improved = True
                break
        if not improved:
            # Try global small perturbation
            for attempt in range(10):
                delta = np.random.randn(N, 2) * 0.0005
                c_new = c + delta
                c_new = np.clip(c_new, -0.499, 0.499)
                r_new = _radius(c_new)
                if r_new > current_r + tol:
                    c = c_new
                    current_r = r_new
                    improved = True
                    break
        if not improved:
            break
    return c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    np.random.seed(42)
    
    # Initial: hexagonal lattice arrangement
    # Approximate hex packing: rows offset
    m = int(np.ceil(np.sqrt(N)))
    # Start with a slightly perturbed grid
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    c = grid[:N].copy()
    # Add small random perturbation to break symmetry
    c += np.random.randn(N, 2) * 0.001
    
    # Force-directed relaxation
    c = _force_directed(c, iterations=2000, lr_start=0.0005, lr_end=0.000005)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return c
    
    # Local polish
    c, best_r = _local_polish(c, max_iters=100, tol=1e-11)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return c
    
    # Final polish with tighter tolerance
    c, best_r2 = _local_polish(c, max_iters=50, tol=1e-12)
    
    # Ensure validity: clip to square
    c = np.clip(c, -0.499, 0.499)
    
    return c
