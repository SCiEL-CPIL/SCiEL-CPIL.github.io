"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.spatial import cKDTree

def _score(c):
    """Compute the largest common radius for centres c (shape (N,3))."""
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    # distance to walls
    wall = np.min(np.min(0.5 - np.abs(c), axis=1), axis=0)
    # pairwise distances
    tree = cKDTree(c)
    # Query nearest neighbors
    # We only need the nearest neighbor for each point
    dists, _ = tree.query(c, k=2)
    min_dist = np.min(dists[:, 1])
    return min(wall, min_dist / 2.0)

def _init_grid():
    """Simple cubic grid as starting point."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _init_random():
    """Random initialisation inside the cube."""
    return np.random.uniform(-0.5, 0.5, size=(N, 3))

def _local_search(c, r, max_iters=2000, time_limit=None, start_time=None):
    """
    Lloyd's algorithm variant:
    1. Move each sphere towards the center of its Voronoi cell (approximated by moving away from neighbors).
    2. Resolve collisions by pushing overlapping spheres apart.
    3. Keep spheres inside the box.
    
    This is a heuristic local search that tries to find a valid packing for radius r.
    """
    c = c.copy()
    # Initial cleanup: if any spheres are outside or overlapping significantly, nudge them
    # We use a simple iterative relaxation
    
    for i in range(max_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
            
        # 1. Compute desired moves based on "pressure"
        # We want to maximize the minimum distance.
        # A simple heuristic: move each point away from its closest neighbors.
        
        tree = cKDTree(c)
        dists, indices = tree.query(c, k=2) # k=2 to get self + 1 neighbor
        
        # dists[:, 0] is 0, dists[:, 1] is distance to nearest neighbor
        # indices[:, 0] is self, indices[:, 1] is nearest neighbor
        
        moves = np.zeros_like(c)
        for idx in range(N):
            neighbor_idx = indices[idx, 1]
            if dists[idx, 1] < 2.0 * r:
                # Overlapping or too close
                diff = c[idx] - c[neighbor_idx]
                norm = np.linalg.norm(diff)
                if norm > 1e-6:
                    # Push away
                    moves[idx] += diff / norm
                else:
                    moves[idx] += np.random.normal(0, 0.01, 3)
            
            # Also check wall constraints
            # If close to wall, push in
            for dim in range(3):
                if c[idx, dim] < -0.5 + r:
                    moves[idx, dim] += 1.0
                elif c[idx, dim] > 0.5 - r:
                    moves[idx, dim] -= 1.0

        # Normalize moves to prevent large steps
        move_norms = np.linalg.norm(moves, axis=1, keepdims=True)
        move_norms = np.where(move_norms < 1e-6, 1.0, move_norms)
        moves = moves / move_norms
        
        # Step size: small to ensure stability
        step_size = 0.05 * r
        
        c += moves * step_size
        
        # Hard constraints: clip to box
        c = np.clip(c, -0.5 + r, 0.5 - r)
        
        # Hard collision resolution (simple iteration)
        # This is expensive if done naively, so we do a few passes
        for _ in range(5):
            tree = cKDTree(c)
            dists, indices = tree.query(c, k=2)
            for idx in range(N):
                neighbor_idx = indices[idx, 1]
                d = dists[idx, 1]
                if d < 2.0 * r:
                    diff = c[idx] - c[neighbor_idx]
                    norm = np.linalg.norm(diff)
                    if norm > 1e-6:
                        # Move both apart by half the overlap
                        overlap = (2.0 * r - d) / 2.0
                        c[idx] += (diff / norm) * overlap
                        c[neighbor_idx] -= (diff / norm) * overlap
            c = np.clip(c, -0.5 + r, 0.5 - r)

        # Check if we have a valid packing
        tree = cKDTree(c)
        dists, _ = tree.query(c, k=2)
        min_dist = np.min(dists[:, 1])
        min_wall = np.min(np.min(0.5 - np.abs(c), axis=1), axis=0)
        if min_dist >= 2.0 * r and min_wall >= r:
            return c, True

    return c, False

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_c = _init_grid()
    best_score = _score(best_c)
    
    # Binary search on radius
    # Lower bound: 0.1 (safe)
    # Upper bound: 0.12 (known best is ~0.111)
    lo = 0.10
    hi = 0.12
    
    # We will try to find the maximum r for which we can find a valid packing.
    # We use a few random restarts for each r trial.
    
    # Pre-compute a good initial guess from grid
    c_grid = _init_grid()
    
    # Try to improve the grid packing with a local search at a moderate radius first
    # to get a better baseline
    c_base, _ = _local_search(c_grid, 0.10, max_iters=500, time_limit=10.0, start_time=start_time)
    sc_base = _score(c_base)
    if sc_base > best_score:
        best_score = sc_base
        best_c = c_base

    # Binary search steps
    # We limit the number of steps to ensure we finish in time
    # Each step involves a local search which can be expensive.
    # We will do about 10-15 iterations of binary search.
    
    num_iters = 12
    for i in range(num_iters):
        if time.time() > deadline - 10.0:
            break
            
        mid = (lo + hi) / 2.0
        
        # Try to find a valid packing for radius 'mid'
        # Use multiple restarts
        found_valid = False
        best_c_for_mid = None
        
        # Restart 1: From current best
        c_try, valid = _local_search(best_c, mid, max_iters=1000, time_limit=10.0, start_time=start_time)
        if valid:
            found_valid = True
            best_c_for_mid = c_try
        else:
            # Restart 2: From grid
            c_try, valid = _local_search(c_grid, mid, max_iters=1000, time_limit=10.0, start_time=start_time)
            if valid:
                found_valid = True
                best_c_for_mid = c_try
            else:
                # Restart 3: Random
                c_rand = _init_random()
                c_try, valid = _local_search(c_rand, mid, max_iters=1000, time_limit=10.0, start_time=start_time)
                if valid:
                    found_valid = True
                    best_c_for_mid = c_try

        if found_valid:
            lo = mid
            best_c = best_c_for_mid
            # Update best score if this valid packing is better
            sc_mid = _score(best_c)
            if sc_mid > best_score:
                best_score = sc_mid
                best_c = best_c
        else:
            hi = mid
            
    # Final polish: try to maximize the radius of the best found configuration
    # using a local search that doesn't enforce a specific r, but maximizes min distance
    # This is similar to the original force-directed but with hard constraints
    
    if time.time() < deadline - 5.0:
        # Simple gradient-free optimization: 
        # Move each point in the direction that increases the minimum distance to its neighbors
        c_polish = best_c.copy()
        for _ in range(500):
            if time.time() > deadline - 2.0:
                break
            tree = cKDTree(c_polish)
            dists, indices = tree.query(c_polish, k=2)
            moves = np.zeros_like(c_polish)
            for idx in range(N):
                neighbor_idx = indices[idx, 1]
                diff = c_polish[idx] - c_polish[neighbor_idx]
                norm = np.linalg.norm(diff)
                if norm > 1e-6:
                    moves[idx] += diff / norm
                # Wall push
                for dim in range(3):
                    if c_polish[idx, dim] < -0.5 + 0.1:
                        moves[idx, dim] += 1.0
                    elif c_polish[idx, dim] > 0.5 - 0.1:
                        moves[idx, dim] -= 1.0
            
            move_norms = np.linalg.norm(moves, axis=1, keepdims=True)
            move_norms = np.where(move_norms < 1e-6, 1.0, move_norms)
            moves = moves / move_norms
            
            c_polish += moves * 0.01
            c_polish = np.clip(c_polish, -0.5, 0.5)
            
            sc_polish = _score(c_polish)
            if sc_polish > best_score:
                best_score = sc_polish
                best_c = c_polish.copy()
            else:
                # If score dropped, revert and reduce step size? 
                # For simplicity, we just keep the best found so far.
                pass

    return best_c
# EVOLVE-BLOCK-END
