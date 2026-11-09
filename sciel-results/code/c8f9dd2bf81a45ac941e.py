"""Rewritten program: physics-based repulsion + local optimization for circle packing."""
import numpy as np
import time

N = 200
HALF = 0.5  # half side of square


def _score(c):
    """Compute the radius for a configuration c (N,2)."""
    if c.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    # wall distances
    wall = np.min(np.min(HALF - np.abs(c), axis=1))
    if wall < 0:
        return -1.0
    # pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_pair, wall)


def _initial_random(seed):
    """Random valid configuration: grid-like jittered start."""
    rng = np.random.RandomState(seed)
    # Use a grid as base then jitter
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N].astype(float)
    # Add small jitter to break symmetry
    jitter = rng.uniform(-0.01, 0.01, size=(N, 2))
    c = grid + jitter
    # Clamp to square
    c = np.clip(c, -HALF + 0.001, HALF - 0.001)
    return c


def _repulsion_step(c, r_target, iters=500, lr=0.001):
    """Iteratively push circles apart using repulsion forces.
    
    For each pair closer than 2*r_target, apply a repulsive force.
    Also push away from walls if closer than r_target.
    """
    c = c.copy()
    for it in range(iters):
        forces = np.zeros_like(c)
        # Pairwise repulsion
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, 1e-10)
        
        # Only consider pairs that are too close
        too_close = dists < 2 * r_target
        # Force magnitude: proportional to overlap
        overlap = np.maximum(0, 2 * r_target - dists)
        # Direction: from j to i (diff is c_i - c_j, so force on i is along diff)
        unit = diff / dists[:, :, np.newaxis]
        # Force on i from j: unit * overlap
        force_mat = unit * overlap[:, :, np.newaxis]
        forces += np.sum(force_mat, axis=1)
        
        # Wall repulsion
        for axis in range(2):
            # Left wall
            dist_left = c[:, axis] + HALF
            overlap_left = np.maximum(0, r_target - dist_left)
            forces[:, axis] -= overlap_left
            # Right wall
            dist_right = HALF - c[:, axis]
            overlap_right = np.maximum(0, r_target - dist_right)
            forces[:, axis] += overlap_right
        
        c += lr * forces
        # Clamp to square
        c = np.clip(c, -HALF + 1e-6, HALF - 1e-6)
    
    return c


def _local_refine(c, r_target, iters=2000, lr=0.0005):
    """Local optimization: gradient ascent on the score function.
    
    We maximize min(pairwise_dist/2, wall_dist).
    Use subgradient method.
    """
    c = c.copy()
    best_r = _score(c)
    
    for it in range(iters):
        # Compute current score and identify active constraints
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        wall = np.min(HALF - np.abs(c), axis=1)
        min_wall = np.min(wall)
        r = min(min_pair, min_wall)
        
        if r > best_r + 1e-12:
            best_r = r
        
        # Subgradient: average of active constraint gradients
        grad = np.zeros_like(c)
        
        if r == min_pair:
            # Find all pairs achieving min distance
            mask = dists <= 2 * min_pair + 1e-10
            # Gradient of dist(i,j) wrt c_i is (c_i - c_j)/dist
            # We want to increase min distance, so move apart
            for i in range(N):
                for j in range(N):
                    if mask[i, j] and i != j:
                        d = dists[i, j]
                        if d > 1e-10:
                            # Move i away from j
                            grad[i] += (c[i] - c[j]) / d
                            # Move j away from i
                            grad[j] -= (c[i] - c[j]) / d
        
        if r == min_wall:
            # Find circles at min wall distance
            for i in range(N):
                for axis in range(2):
                    if abs(wall[i] - min_wall) < 1e-10:
                        # Determine which wall
                        if c[i, axis] >= 0:
                            # Right/top wall: move inward
                            grad[i, axis] -= 1
                        else:
                            # Left/bottom wall: move inward
                            grad[i, axis] += 1
        
        # Normalize gradient
        norm = np.linalg.norm(grad)
        if norm > 1e-10:
            grad /= norm
        
        c += lr * grad
        c = np.clip(c, -HALF + 1e-6, HALF - 1e-6)
    
    return c


def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Start from jittered grid
    c = _initial_random(42)
    r0 = _score(c)
    best_c = c.copy()
    best_r = r0
    
    # Phase 1: Repulsion to find a good radius
    # Estimate a target radius slightly above current
    r_target = r0 * 1.1
    
    while time.time() < deadline - 30:
        c = _repulsion_step(c, r_target, iters=200, lr=0.002)
        r_new = _score(c)
        if r_new > best_r:
            best_r = r_new
            best_c = c.copy()
            r_target = r_new * 1.05
        else:
            # If no improvement, try slightly lower target
            r_target = max(r0, r_target * 0.98)
            if r_target <= r0 * 0.95:
                break
    
    # Phase 2: Local refinement
    while time.time() < deadline - 10:
        c = _local_refine(best_c, best_r, iters=500, lr=0.0003)
        r_new = _score(c)
        if r_new > best_r:
            best_r = r_new
            best_c = c.copy()
        else:
            break
    
    # Final check
    final_r = _score(best_c)
    if final_r > best_r:
        best_r = final_r
    
    return best_c
