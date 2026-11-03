"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time

def _compute_radius(c):
    """
    Compute the exact maximum radius for the given centers.
    r = min( min_{i<j} ||c_i - c_j|| / 2, min_i min_k (0.5 - |c_ik|) )
    """
    if c.shape[0] < 2:
        return 0.5 - np.max(np.abs(c))
    
    # Wall clearance
    wall = 0.5 - np.max(np.abs(c))
    
    # Pairwise distances
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    
    return min(wall, min_pair)

def _generate_initial_configs(n_configs=6, seed=42):
    """
    Generate diverse initial configurations using different lattice structures
    and random perturbations.
    """
    rng = np.random.default_rng(seed)
    configs = []
    
    # BCC lattice
    n_div = 5
    bcc_points = []
    for i in range(n_div + 1):
        for j in range(n_div + 1):
            for k in range(n_div + 1):
                bcc_points.append([i / n_div, j / n_div, k / n_div])
    for i in range(n_div):
        for j in range(n_div):
            for k in range(n_div):
                bcc_points.append([(i + 0.5) / n_div, (j + 0.5) / n_div, (k + 0.5) / n_div])
    bcc_points = np.array(bcc_points) - 0.5
    
    # FCC lattice
    n_div_fcc = 4
    fcc_points = []
    for i in range(n_div_fcc + 1):
        for j in range(n_div_fcc + 1):
            for k in range(n_div_fcc + 1):
                fcc_points.append([i / n_div_fcc, j / n_div_fcc, k / n_div_fcc])
    for i in range(n_div_fcc):
        for j in range(n_div_fcc + 1):
            for k in range(n_div_fcc + 1):
                fcc_points.append([(i + 0.5) / n_div_fcc, j / n_div_fcc, k / n_div_fcc])
    for i in range(n_div_fcc + 1):
        for j in range(n_div_fcc):
            for k in range(n_div_fcc + 1):
                fcc_points.append([i / n_div_fcc, (j + 0.5) / n_div_fcc, k / n_div_fcc])
    for i in range(n_div_fcc + 1):
        for j in range(n_div_fcc + 1):
            for k in range(n_div_fcc):
                fcc_points.append([i / n_div_fcc, j / n_div_fcc, (k + 0.5) / n_div_fcc])
    fcc_points = np.array(fcc_points) - 0.5
    mask = np.all(np.abs(fcc_points) <= 0.5, axis=1)
    fcc_points = fcc_points[mask]
    
    def select_from_lattice(points, n_select, min_dist, rng_local):
        indices = np.arange(len(points))
        rng_local.shuffle(indices)
        points_shuffled = points[indices]
        selected = [points_shuffled[0]]
        for i in range(1, len(points_shuffled)):
            if len(selected) >= n_select:
                break
            p = points_shuffled[i]
            dists = np.linalg.norm(np.array(selected) - p, axis=1)
            if np.min(dists) > min_dist:
                selected.append(p)
        while len(selected) < n_select:
            p = rng_local.uniform(-0.45, 0.45, size=3)
            selected.append(p)
        return np.array(selected[:n_select])
    
    # Generate configs from BCC
    for noise_scale in [0.0, 0.01, 0.02]:
        base = select_from_lattice(bcc_points, N, 0.08, np.random.default_rng(100 + int(noise_scale*1000)))
        if noise_scale > 0:
            base = base + np.random.default_rng(200 + int(noise_scale*1000)).normal(0, noise_scale, size=(N, 3))
            base = np.clip(base, -0.45, 0.45)
        configs.append(base.copy())
    
    # Generate configs from FCC
    for noise_scale in [0.0, 0.01]:
        base = select_from_lattice(fcc_points, N, 0.09, np.random.default_rng(300 + int(noise_scale*1000)))
        if noise_scale > 0:
            base = base + np.random.default_rng(400 + int(noise_scale*1000)).normal(0, noise_scale, size=(N, 3))
            base = np.clip(base, -0.45, 0.45)
        configs.append(base.copy())
    
    # Random grid-based config
    grid_size = 10
    ticks = (np.arange(grid_size) + 0.5) / grid_size - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid_indices = np.arange(len(grid))
    rng.shuffle(grid_indices)
    grid_shuffled = grid[grid_indices]
    selected = [grid_shuffled[0]]
    for i in range(1, len(grid_shuffled)):
        if len(selected) >= N:
            break
        p = grid_shuffled[i]
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if np.min(dists) > 0.1:
            selected.append(p)
    while len(selected) < N:
        p = grid_shuffled[rng.integers(len(grid_shuffled))]
        dists = np.linalg.norm(np.array(selected) - p, axis=1)
        if len(dists) == 0 or np.min(dists) > 0.05:
            selected.append(p)
        else:
            p = rng.uniform(-0.45, 0.45, size=3)
            selected.append(p)
    configs.append(np.array(selected[:N]))
    
    return configs

def _simulated_annealing(c_init, deadline, seed=0):
    """
    Simulated annealing to maximize the minimum radius.
    Uses a temperature schedule and accepts/rejects moves based on Metropolis criterion.
    """
    rng = np.random.default_rng(seed)
    current_c = c_init.copy()
    current_r = _compute_radius(current_c)
    best_c = current_c.copy()
    best_r = current_r
    
    # Temperature schedule
    T_start = 0.05
    T_end = 1e-7
    cooling_rate = 0.995
    
    T = T_start
    step_size = 0.02
    
    while time.time() < deadline and T > T_end:
        # Generate a random move: pick a random sphere and move it
        i = rng.integers(N)
        delta = rng.normal(0, step_size, size=3)
        
        c_trial = current_c.copy()
        c_trial[i] += delta
        c_trial = np.clip(c_trial, -0.499999, 0.499999)
        
        trial_r = _compute_radius(c_trial)
        delta_r = trial_r - current_r
        
        # Metropolis criterion: accept if better, or with probability exp(-delta_r/T) if worse
        if delta_r >= 0 or rng.random() < np.exp(-delta_r / T):
            current_c = c_trial
            current_r = trial_r
            
            if current_r > best_r:
                best_r = current_r
                best_c = current_c.copy()
        
        # Cool down
        T *= cooling_rate
        # Gradually reduce step size
        step_size = max(0.001, step_size * 0.999)
    
    return best_c, best_r

def _local_refinement(c, deadline):
    """
    Local refinement using gradient-free coordinate descent.
    Tries small perturbations in each coordinate to improve the radius.
    """
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    
    step_sizes = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    
    for step in step_sizes:
        if time.time() > deadline - 1:
            break
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for i in range(N):
                for k in range(3):
                    for sign in [-1, 1]:
                        c_trial = best_c.copy()
                        c_trial[i, k] += sign * step
                        c_trial[i, k] = np.clip(c_trial[i, k], -0.499999, 0.499999)
                        
                        trial_r = _compute_radius(c_trial)
                        if trial_r > best_r + 1e-12:
                            best_r = trial_r
                            best_c = c_trial
                            improved = True
    
    return best_c, best_r

def _pair_separation(c, deadline):
    """
    Identify the most constrained pair and try to separate them.
    """
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    
    d = cdist(best_c, best_c)
    np.fill_diagonal(d, np.inf)
    min_pair_idx = np.unravel_index(np.argmin(d), d.shape)
    min_pair_val = d[min_pair_idx] / 2.0
    
    wall = 0.5 - np.abs(best_c)
    min_wall_val = np.min(wall)
    
    if min_pair_val < min_wall_val:
        i, j = min_pair_idx
        diff = best_c[i] - best_c[j]
        norm = np.linalg.norm(diff)
        if norm > 1e-10:
            direction = diff / norm
            for delta in [0.001, 0.0005, 0.0001]:
                c_trial = best_c.copy()
                c_trial[i] += direction * delta
                c_trial[j] -= direction * delta
                c_trial = np.clip(c_trial, -0.499999, 0.499999)
                trial_r = _compute_radius(c_trial)
                if trial_r > best_r + 1e-12:
                    best_r = trial_r
                    best_c = c_trial
                    break
    
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Generate initial configurations
    configs = _generate_initial_configs(n_configs=6, seed=42)
    
    best_c = configs[0].copy()
    best_r = _compute_radius(best_c)
    
    # Run simulated annealing from each initial config
    for idx, c_init in enumerate(configs):
        if time.time() > deadline - 30:
            break
        
        try:
            c_sa, r_sa = _simulated_annealing(c_init, deadline - 30, seed=1000 + idx)
            if r_sa > best_r:
                best_r = r_sa
                best_c = c_sa.copy()
        except Exception:
            pass
    
    # Local refinement
    if time.time() < deadline - 15:
        best_c, best_r = _local_refinement(best_c, deadline - 15)
    
    # Pair separation
    if time.time() < deadline - 10:
        best_c, best_r = _pair_separation(best_c, deadline - 10)
    
    # Final fine-tuning with very small steps
    if time.time() < deadline - 5:
        step_sizes = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
        for step in step_sizes:
            if time.time() > deadline - 2:
                break
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                for i in range(N):
                    for k in range(3):
                        for sign in [-1, 1]:
                            c_trial = best_c.copy()
                            c_trial[i, k] += sign * step
                            c_trial[i, k] = np.clip(c_trial[i, k], -0.499999, 0.499999)
                            trial_r = _compute_radius(c_trial)
                            if trial_r > best_r + 1e-12:
                                best_r = trial_r
                                best_c = c_trial
                                improved = True
    
    # Ensure valid output
    best_c = np.clip(best_c, -0.499999999, 0.499999999)
    best_c = np.asarray(best_c, dtype=float)
    
    if best_c.shape != (N, 3) or not np.all(np.isfinite(best_c)):
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_c = grid[:N]
    
    # Final validation: ensure no coincident centers
    d = cdist(best_c, best_c)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 1e-10:
        rng_fix = np.random.default_rng(999)
        for i in range(N):
            for j in range(i+1, N):
                if np.linalg.norm(best_c[i] - best_c[j]) < 1e-10:
                    best_c[j] += rng_fix.uniform(-1e-8, 1e-8, size=3)
        best_c = np.clip(best_c, -0.499999999, 0.499999999)
    
    return best_c

# EVOLVE-BLOCK-END
