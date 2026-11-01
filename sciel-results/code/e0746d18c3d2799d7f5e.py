"""Combined approach: force-directed repulsion with greedy local search refinement."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

N = 100

# EVOLVE-BLOCK-START

def _compute_r(centres):
    """Compute the maximum valid radius for given centres in [-0.5, 0.5]^3."""
    if not np.all(np.isfinite(centres)):
        return 0.0
    if centres.shape != (N, 3):
        return 0.0
    # Wall constraint: distance from each centre to the nearest wall
    wall_dists = np.min(0.5 - np.abs(centres), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall < 0:
        return 0.0
    # Pairwise distances using cKDTree for efficiency
    tree = cKDTree(centres)
    pairs = tree.query_pairs(1.0, output_type='ndarray')
    if len(pairs) == 0:
        return min_wall
    diffs = centres[pairs[:, 0]] - centres[pairs[:, 1]]
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    min_pair = np.min(dists)
    return min(min_pair / 2.0, min_wall)


def _force_refine(centres, max_iters=400, dt_init=0.015):
    """Force-directed repulsion with wall constraints and adaptive step size."""
    centres = centres.copy().astype(np.float64)
    best_centres = centres.copy()
    best_r = _compute_r(best_centres)
    
    for it in range(max_iters):
        # Compute forces using cKDTree for efficiency
        tree = cKDTree(centres)
        pairs = tree.query_pairs(0.4, output_type='ndarray')
        
        forces = np.zeros_like(centres)
        
        if len(pairs) > 0:
            for i, j in pairs:
                diff = centres[i] - centres[j]
                dist = np.sqrt(np.sum(diff**2))
                if dist < 1e-12:
                    diff = np.random.randn(3) * 1e-6
                    dist = np.sqrt(np.sum(diff**2))
                # Strong repulsive force when close
                force_mag = 1.0 / (dist**2 + 1e-8)
                forces[i] += force_mag * diff / dist
                forces[j] -= force_mag * diff / dist
        
        # Wall forces: push away from walls
        for axis in range(3):
            over = centres[:, axis] - 0.5
            mask_pos = over > -0.05
            if np.any(mask_pos):
                forces[mask_pos, axis] -= 50.0 * np.clip(over[mask_pos], 0, None) + 1.0 * np.clip(0.05 + over[mask_pos], 0, None)
            under = centres[:, axis] + 0.5
            mask_neg = under < 0.05
            if np.any(mask_neg):
                forces[mask_neg, axis] += 50.0 * np.clip(-under[mask_neg], 0, None) + 1.0 * np.clip(0.05 - under[mask_neg], 0, None)
        
        # Update positions with adaptive step size
        dt = dt_init * (0.997 ** it)
        centres += dt * forces
        
        # Clamp to valid region
        centres = np.clip(centres, -0.499, 0.499)
        
        # Evaluate
        r = _compute_r(centres)
        if r > best_r:
            best_r = r
            best_centres = centres.copy()
    
    return best_centres, best_r


def _greedy_local_search(centres, deadline, rng):
    """
    Greedy local search with random kicks, inspired by inspiration 1.
    Moves one particle at a time, accepting improvements.
    """
    c = centres.copy()
    best_c = c.copy()
    best_r = _compute_r(c)
    
    step_size = 0.008
    kick_size = 0.06
    no_improve_count = 0
    max_no_improve = 1500
    
    while time.time() < deadline:
        # Pick a random particle
        idx = rng.integers(0, N)
        
        # Try to move it in a random direction
        c_new = c.copy()
        c_new[idx] += rng.normal(0, step_size, 3)
        c_new = np.clip(c_new, -0.5, 0.5)
        
        r_new = _compute_r(c_new)
        
        if r_new > best_r + 1e-12:
            c = c_new
            best_c = c_new.copy()
            best_r = r_new
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # If stuck, do a random kick to escape local optimum
        if no_improve_count > max_no_improve:
            kick_indices = rng.choice(N, size=12, replace=False)
            c[kick_indices] += rng.normal(0, kick_size, (12, 3))
            c = np.clip(c, -0.5, 0.5)
            r_curr = _compute_r(c)
            if r_curr > best_r:
                best_c = c.copy()
                best_r = r_curr
            no_improve_count = 0
            step_size = 0.01
    
    return best_c, best_r


def _local_optimize(centres, max_iters=150):
    """Use scipy minimize to maximize the minimum radius."""
    def neg_radius(x):
        c = x.reshape(N, 3)
        r = _compute_r(c)
        return -r
    
    x0 = centres.flatten()
    result = minimize(neg_radius, x0, method='Nelder-Mead',
                      options={'maxiter': max_iters, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
    
    c_opt = result.x.reshape(N, 3)
    r_opt = -result.fun
    return c_opt, r_opt


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Start with a simple grid initialization
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centres = grid[:N].copy()
    
    best_centres = centres.copy()
    best_r = _compute_r(best_centres)
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts with different initializations
    n_restarts = 4
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Create perturbed initialization
        if restart == 0:
            init = centres.copy()
        elif restart == 1:
            init = centres.copy() + rng.normal(0, 0.02, (N, 3))
            init = np.clip(init, -0.49, 0.49)
        elif restart == 2:
            init = rng.uniform(-0.45, 0.45, (N, 3))
        else:
            # Perturb the best found so far
            init = best_centres.copy() + rng.normal(0, 0.04, (N, 3))
            init = np.clip(init, -0.49, 0.49)
        
        # Phase 1: Force-directed refinement
        remaining = deadline - time.time()
        if remaining < 15.0:
            break
        
        c_refined, r_refined = _force_refine(init, max_iters=250)
        if r_refined > best_r:
            best_r = r_refined
            best_centres = c_refined.copy()
        
        if time.time() > deadline - 15.0:
            break
        
        # Phase 2: Greedy local search for fine-tuning
        remaining = deadline - time.time()
        if remaining < 10.0:
            break
        
        # Allocate remaining time between local search and optimization
        ls_deadline = time.time() + min(remaining * 0.5, 30.0)
        ls_deadline = min(ls_deadline, deadline - 10.0)
        
        c_ls, r_ls = _greedy_local_search(c_refined, ls_deadline, rng)
        if r_ls > best_r:
            best_r = r_ls
            best_centres = c_ls.copy()
        
        if time.time() > deadline - 8.0:
            break
        
        # Phase 3: Nelder-Mead local optimization
        c_opt, r_opt = _local_optimize(best_centres, max_iters=100)
        if r_opt > best_r:
            best_r = r_opt
            best_centres = c_opt.copy()
    
    # Final validation
    final_r = _compute_r(best_centres)
    if final_r > best_r:
        best_r = final_r
    
    return best_centres

# EVOLVE-BLOCK-END
