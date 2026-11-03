"""Improved program for packing 100 equal spheres in a unit cube using simulated annealing."""
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import random
import math

def _compute_radius(c):
    """
    Compute the maximum common radius for a configuration.
    r = min(min_pairwise_dist / 2, min_boundary_dist)
    """
    # Boundary distance
    abs_c = np.abs(c)
    boundary_dists = np.min(0.5 - abs_c, axis=1)
    r_bound = np.min(boundary_dists)
    
    # Pairwise distances - use vectorized computation
    # For N=100, we have 4950 pairs
    iu = np.triu_indices(N, k=1)
    diff = c[iu[0]] - c[iu[1]]
    dists = np.sqrt(np.sum(diff**2, axis=1))
    r_pair = np.min(dists) / 2.0
    
    return min(r_pair, r_bound)

def _get_closest_pair(c):
    """Return indices of the closest pair."""
    iu = np.triu_indices(N, k=1)
    diff = c[iu[0]] - c[iu[1]]
    dists = np.sqrt(np.sum(diff**2, axis=1))
    min_idx = np.argmin(dists)
    return iu[0][min_idx], iu[1][min_idx]

def _get_boundary_violator(c):
    """Return index of point closest to boundary and which axis."""
    abs_c = np.abs(c)
    # Distance to each of 6 faces
    d = np.column_stack([
        0.5 - abs_c[:, 0],  # x = -0.5 face
        0.5 - abs_c[:, 1],  # y = -0.5 face
        0.5 - abs_c[:, 2],  # z = -0.5 face
    ])
    min_val = np.min(d)
    flat_idx = np.argmin(d.flatten())
    point_idx = flat_idx // 3
    axis = flat_idx % 3
    return point_idx, axis

def _stochastic_local_search(c, time_budget=60):
    """
    Stochastic local search with simulated annealing.
    Uses swaps and small perturbations to escape local minima.
    """
    import time
    
    c = c.copy()
    current_r = _compute_radius(c)
    best_c = c.copy()
    best_r = current_r
    
    start_time = time.time()
    T = 0.008
    T_min = 1e-8
    cooling = 0.9995
    
    # Precompute upper triangle for fast distance updates
    iu = np.triu_indices(N, k=1)
    
    iteration = 0
    while time.time() - start_time < time_budget:
        iteration += 1
        
        # Choose move type
        move_type = random.random()
        
        if move_type < 0.3:
            # Single point perturbation
            i = random.randint(0, N - 1)
            c_new = c.copy()
            step = 0.01 * (1 + 0.5 * np.exp(-iteration / 5000))
            c_new[i] += np.random.randn(3) * step
            c_new[i] = np.clip(c_new[i], -0.5, 0.5)
            
        elif move_type < 0.5:
            # Two point perturbation
            i, j = random.sample(range(N), 2)
            c_new = c.copy()
            step = 0.008 * (1 + 0.5 * np.exp(-iteration / 5000))
            c_new[i] += np.random.randn(3) * step
            c_new[j] += np.random.randn(3) * step
            c_new[i] = np.clip(c_new[i], -0.5, 0.5)
            c_new[j] = np.clip(c_new[j], -0.5, 0.5)
            
        elif move_type < 0.7:
            # Swap two points
            i, j = random.sample(range(N), 2)
            c_new = c.copy()
            c_new[i], c_new[j] = c_new[j].copy(), c_new[i].copy()
            
        elif move_type < 0.85:
            # Push closest pair apart
            i, j = _get_closest_pair(c)
            c_new = c.copy()
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = diff / dist
            
            # Move apart with some randomness
            step = 0.003 + 0.002 * random.random()
            c_new[i] += direction * step + np.random.randn(3) * 0.001
            c_new[j] -= direction * step + np.random.randn(3) * 0.001
            c_new[i] = np.clip(c_new[i], -0.5, 0.5)
            c_new[j] = np.clip(c_new[j], -0.5, 0.5)
            
        else:
            # Pull boundary violator inward
            i, axis = _get_boundary_violator(c)
            c_new = c.copy()
            sign = 1 if c[i, axis] >= 0 else -1
            step = 0.003 + 0.002 * random.random()
            c_new[i, axis] += sign * step
            c_new[i] = np.clip(c_new[i], -0.5, 0.5)
            # Add small random perturbation
            c_new[i] += np.random.randn(3) * 0.002
            c_new[i] = np.clip(c_new[i], -0.5, 0.5)
        
        new_r = _compute_radius(c_new)
        
        # Acceptance criterion
        if new_r >= current_r:
            c = c_new
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_c = c.copy()
        else:
            # Accept worse moves with probability exp((new_r - current_r) / T)
            delta = new_r - current_r
            prob = math.exp(delta / T)
            if random.random() < prob:
                c = c_new
                current_r = new_r
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Occasionally restart temperature to escape local minima
        if iteration % 5000 == 0 and T < 0.001:
            T = 0.003
    
    return best_c, best_r

def _fast_refine(c, max_iters=2000):
    """
    Fast greedy refinement: iteratively push the most constrained points.
    """
    c = c.copy()
    current_r = _compute_radius(c)
    best_c = c.copy()
    best_r = current_r
    
    for it in range(max_iters):
        # Compute current radius components
        abs_c = np.abs(c)
        boundary_dists = np.min(0.5 - abs_c, axis=1)
        r_bound = np.min(boundary_dists)
        
        iu = np.triu_indices(N, k=1)
        diff = c[iu[0]] - c[iu[1]]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        r_pair = np.min(dists) / 2.0
        
        if r_pair <= r_bound:
            # Push closest pair apart
            min_idx = np.argmin(dists)
            i, j = iu[0][min_idx], iu[1][min_idx]
            
            diff_ij = c[i] - c[j]
            dist_ij = np.linalg.norm(diff_ij)
            if dist_ij < 1e-10:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            else:
                direction = diff_ij / dist_ij
            
            c_new = c.copy()
            step = 0.002
            c_new[i] = np.clip(c_new[i] + direction * step, -0.5, 0.5)
            c_new[j] = np.clip(c_new[j] - direction * step, -0.5, 0.5)
            
            new_r = _compute_radius(c_new)
            if new_r > current_r:
                c = c_new
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_c = c.copy()
        else:
            # Push boundary violator inward
            i, axis = _get_boundary_violator(c)
            c_new = c.copy()
            sign = 1 if c[i, axis] >= 0 else -1
            c_new[i, axis] = np.clip(c_new[i, axis] + sign * 0.002, -0.5, 0.5)
            
            new_r = _compute_radius(c_new)
            if new_r > current_r:
                c = c_new
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_c = c.copy()
        
        # Small random kick every 200 iterations
        if it % 200 == 0:
            idx = random.randint(0, N - 1)
            c_new = c.copy()
            c_new[idx] += np.random.randn(3) * 0.005
            c_new[idx] = np.clip(c_new[idx], -0.5, 0.5)
            new_r = _compute_radius(c_new)
            if new_r > current_r:
                c = c_new
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_c = c.copy()
    
    return best_c, best_r

def _initial_placement():
    """
    Create a good initial placement using a jittered grid.
    """
    # Use a 5x5x4 grid = 100 points
    # Place them in the cube [-0.5, 0.5]^3
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.5 + 0.1, 0.5 - 0.1, nx)
    ys = np.linspace(-0.5 + 0.1, 0.5 - 0.1, ny)
    zs = np.linspace(-0.5 + 0.1, 0.5 - 0.1, nz)
    
    points = []
    for z in zs:
        for y in ys:
            for x in xs:
                points.append([x, y, z])
    
    c = np.array(points[:N])
    
    # Add small random jitter
    c += np.random.randn(N, 3) * 0.01
    c = np.clip(c, -0.5, 0.5)
    
    return c

def construct():
    """Construct the optimal arrangement of 100 equal spheres in a unit cube."""
    # Start with a jittered grid
    c = _initial_placement()
    
    # Initial refinement
    c, r = _fast_refine(c, max_iters=1000)
    
    # Stochastic local search - main optimization
    c, r = _stochastic_local_search(c, time_budget=45)
    
    # Final refinement
    c, r2 = _fast_refine(c, max_iters=2000)
    if r2 > r:
        r = r2
    
    # One more round of local search with lower temperature
    c, r3 = _stochastic_local_search(c, time_budget=15)
    if r3 > r:
        r = r3
        c, r4 = _fast_refine(c, max_iters=1000)
        if r4 > r:
            r = r4
    
    return c
# EVOLVE-BLOCK-END
