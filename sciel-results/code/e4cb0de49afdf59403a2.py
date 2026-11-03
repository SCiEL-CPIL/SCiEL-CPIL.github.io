# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import cKDTree
import time
import random

N = 100
CUBE_HALF = 0.5

def _score(c):
    """
    Compute the maximum radius for a given set of centers.
    r = min(min_pairwise_dist / 2, min_boundary_dist)
    """
    if N > 1:
        dists = pdist(c, metric='euclidean')
        min_pair_dist = np.min(dists)
    else:
        min_pair_dist = np.inf
        
    abs_c = np.abs(c)
    boundary_dists = CUBE_HALF - np.max(abs_c, axis=1)
    min_boundary_dist = np.min(boundary_dists)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def _initial_fcc():
    """
    Create an FCC-like initial configuration.
    FCC packing has density ~0.74. We place points on an FCC lattice
    within the cube, then select 100 points.
    """
    # FCC lattice: points at (i, j, k) where i+j+k is even, scaled appropriately
    # In a unit cube, we can use a grid with spacing s
    # Let's try multiple spacings and pick the best initial score
    
    best_c = None
    best_s = -1
    
    # Try different lattice spacings
    for s in [0.18, 0.19, 0.20, 0.21, 0.22]:
        # Generate FCC points
        points = []
        # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in units of s
        # Actually, let's use a simpler approach: BCC or just a dense grid
        pass
    
    # Simpler: use a 5x5x4 grid with jitter
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Take first 100
    c = grid[:N].copy()
    # Add small random jitter to break symmetry
    c += np.random.randn(N, 3) * 0.005
    c = np.clip(c, -CUBE_HALF, CUBE_HALF)
    
    return c

def _initial_random():
    """Create a random initial placement."""
    return np.random.uniform(-CUBE_HALF, CUBE_HALF, size=(N, 3))

def _simulated_annealing(c, time_limit=50, T_start=0.015, T_end=1e-7, alpha=None):
    """
    Improved Simulated Annealing with:
    - Dynamic step size based on current radius
    - Neighbor-aware moves (focus on most constrained points)
    - Multi-phase cooling
    """
    start_time = time.time()
    c = c.copy()
    current_score = _score(c)
    best_c = c.copy()
    best_score = current_score
    
    # Determine number of iterations we can afford
    # Estimate: ~10000-50000 iterations per second depending on overhead
    # We'll just run until time limit
    
    T = T_start
    T_min = T_end
    
    # Adaptive alpha: cool down faster initially, slower later
    # Use a time-based cooling: T = T_start * (T_end/T_start)^(t/T_total)
    # But we don't know T_total exactly, so use a heuristic
    
    # Phase 1: Fast cooling (first 60% of time)
    # Phase 2: Slow cooling (last 40% of time)
    
    # Move probabilities
    p_single = 0.5
    p_double = 0.3
    p_neighbor = 0.2  # Move a point toward/away from its nearest neighbor
    
    # Initial step size
    base_step = 0.03
    
    # Track iteration count for adaptive cooling
    iter_count = 0
    
    while time.time() - start_time < time_limit:
        iter_count += 1
        
        # Adaptive cooling: use exponential decay
        # After 1000 iterations, T should be ~T_start * 0.5
        # After 5000 iterations, T should be ~T_start * 0.1
        # T = T_start * exp(-iter_count / tau)
        # Choose tau so that at time_limit, T ~ T_min
        # Rough estimate: ~20000 iterations in time_limit
        # T_end = T_start * exp(-20000/tau) => tau = -20000/ln(T_end/T_start)
        # ln(T_end/T_start) = ln(1e-7/0.015) = ln(6.67e-6) ≈ -11.9
        # tau ≈ 20000/11.9 ≈ 1680
        tau = 1500
        T = max(T_start * np.exp(-iter_count / tau), T_min)
        
        # Step size scales with T, but also with current radius
        # Larger radius => can afford larger steps
        # r is roughly current_score
        step_scale = max(0.3, min(1.0, current_score / 0.1))
        step_size = base_step * step_scale * (T / T_start + 0.1)
        step_size = max(0.002, min(0.05, step_size))
        
        # Determine move type
        r = random.random()
        c_new = c.copy()
        
        if r < p_single:
            # Move one point - prefer points with smaller boundary distance
            # or points that are part of the minimum distance pair
            # For simplicity, pick randomly but with slight bias
            idx = random.randint(0, N - 1)
            delta = np.random.randn(3) * step_size
            c_new[idx] = c_new[idx] + delta
            c_new[idx] = np.clip(c_new[idx], -CUBE_HALF, CUBE_HALF)
            
        elif r < p_single + p_double:
            # Move two points
            i1, i2 = random.sample(range(N), 2)
            delta1 = np.random.randn(3) * step_size
            delta2 = np.random.randn(3) * step_size
            c_new[i1] = c_new[i1] + delta1
            c_new[i2] = c_new[i2] + delta2
            c_new[i1] = np.clip(c_new[i1], -CUBE_HALF, CUBE_HALF)
            c_new[i2] = np.clip(c_new[i2], -CUBE_HALF, CUBE_HALF)
            
        elif r < p_single + p_double + p_neighbor:
            # Neighbor-aware move: pick a point and move it away from its nearest neighbor
            idx = random.randint(0, N - 1)
            # Find nearest neighbor (approximate: just pick a random other point)
            # For efficiency, use a random other point as proxy
            other = random.randint(0, N - 1)
            if other == idx:
                other = (other + 1) % N
            # Direction away from neighbor
            dir_vec = c[idx] - c[other]
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-10:
                dir_vec = dir_vec / norm
            else:
                dir_vec = np.random.randn(3)
                dir_vec = dir_vec / np.linalg.norm(dir_vec)
            
            # Move in direction away from neighbor, plus some random component
            delta = dir_vec * step_size * 0.5 + np.random.randn(3) * step_size * 0.5
            c_new[idx] = c_new[idx] + delta
            c_new[idx] = np.clip(c_new[idx], -CUBE_HALF, CUBE_HALF)
            
        else:
            # Large jump for one point
            idx = random.randint(0, N - 1)
            c_new[idx] = np.random.uniform(-CUBE_HALF, CUBE_HALF, size=3)
        
        new_score = _score(c_new)
        
        # Acceptance
        if new_score >= current_score:
            c = c_new
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_c = c.copy()
        else:
            delta_score = new_score - current_score
            # Avoid division by zero or very small T
            T_safe = max(T, 1e-10)
            prob = np.exp(delta_score / T_safe)
            if random.random() < prob:
                c = c_new
                current_score = new_score
        
    return best_c, best_score

def _local_polish(c, time_limit=10):
    """
    Local polishing: try small moves in 26 directions for each point.
    """
    start_time = time.time()
    c = c.copy()
    current_score = _score(c)
    
    # Directions: 6 axis-aligned + 12 face-diagonal + 8 corner-diagonal
    directions = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                directions.append(np.array([dx, dy, dz], dtype=float))
    directions = np.array(directions)
    
    step = 0.003
    
    while time.time() - start_time < time_limit:
        improved = False
        # Shuffle point order to avoid bias
        order = np.random.permutation(N)
        for i in order:
            # Try all directions
            for d in directions:
                c_test = c.copy()
                c_test[i] = c[i] + d * step
                c_test[i] = np.clip(c_test[i], -CUBE_HALF, CUBE_HALF)
                s_test = _score(c_test)
                if s_test > current_score:
                    c = c_test
                    current_score = s_test
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Try smaller step
            step *= 0.7
            if step < 1e-5:
                break
    
    return c, current_score

def construct():
    """
    Construct the optimal arrangement of 100 equal spheres in a unit cube.
    """
    best_c = None
    best_score = -np.inf
    
    # Time budget: 120s total. Leave 5s for overhead and polishing.
    
    # 1. Grid start with SA
    c_grid = _initial_fcc()
    c_best, s_best = _simulated_annealing(c_grid, time_limit=25, T_start=0.02)
    if s_best > best_score:
        best_score = s_best
        best_c = c_best
        
    # 2. Random starts
    for i in range(3):
        c_rand = _initial_random()
        c_curr, s_curr = _simulated_annealing(c_rand, time_limit=20, T_start=0.025)
        if s_curr > best_score:
            best_score = s_curr
            best_c = c_curr
            
    # 3. Polish the best solution
    if best_c is not None:
        c_polish, s_polish = _local_polish(best_c, time_limit=15)
        if s_polish > best_score:
            best_score = s_polish
            best_c = c_polish
    
    # 4. Final short SA on the polished solution
    if best_c is not None:
        c_final, s_final = _simulated_annealing(best_c, time_limit=10, T_start=0.005, T_end=1e-8)
        if s_final > best_score:
            best_score = s_final
            best_c = c_final
    
    return best_c
# EVOLVE-BLOCK-END
