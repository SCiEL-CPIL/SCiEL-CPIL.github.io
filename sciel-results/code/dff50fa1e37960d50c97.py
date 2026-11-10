# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _neg_log_ratio(flat_pts, n=16):
    """
    Objective function for local optimization.
    Minimizes -log(d_min/d_max) which is equivalent to maximizing log(d_min/d_max).
    This is smoother than the raw ratio for gradient-free methods like Powell.
    """
    pts = flat_pts.reshape(n, 2)
    d = pdist(pts)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 1e10
    ratio = d_min / d_max
    if ratio <= 0:
        return 1e10
    return -np.log(ratio)

def _local_refine_powell(points, max_iter=100):
    """
    Refine points using Powell's method to maximize the min/max distance ratio.
    """
    n = points.shape[0]
    flat = points.flatten()
    
    try:
        # Powell is derivative-free and handles non-smoothness better than L-BFGS-B
        result = minimize(
            _neg_log_ratio,
            flat,
            method='Powell',
            args=(n,),
            options={'maxiter': max_iter, 'xtol': 1e-10, 'ftol': 1e-12}
        )
        opt_points = result.x.reshape(n, 2)
        return opt_points
    except Exception:
        return points

def _generate_initial_configs(n=16):
    """Generate diverse initial configurations."""
    rng = np.random.default_rng(42)
    configs = []
    
    # 1. 4x4 Grid
    grid = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
    grid = (grid - grid.mean(axis=0)) / (grid.std() + 1e-10)
    configs.append(grid)
    
    # 2. Hexagonal Lattice
    hex_pts = []
    for row in range(4):
        for col in range(4):
            x = col + (0.5 if row % 2 else 0)
            y = row * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    hex_pts = (hex_pts - hex_pts.mean(axis=0)) / (hex_pts.std() + 1e-10)
    configs.append(hex_pts)
    
    # 3. Concentric Circles
    # Inner 8, Outer 8
    angles_inner = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles_outer = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
    r_inner = 0.5
    r_outer = 1.5
    pts_inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    pts_outer = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
    pts = np.vstack([pts_inner, pts_outer])
    pts = (pts - pts.mean(axis=0)) / (pts.std() + 1e-10)
    configs.append(pts)
    
    # 4. Random Uniform
    for _ in range(5):
        pts = rng.uniform(-1, 1, size=(n, 2))
        pts = (pts - pts.mean(axis=0)) / (pts.std() + 1e-10)
        configs.append(pts)
        
    # 5. Random Gaussian
    for _ in range(5):
        pts = rng.standard_normal((n, 2))
        pts = (pts - pts.mean(axis=0)) / (pts.std() + 1e-10)
        configs.append(pts)

    return configs

def _simulated_annealing(points, start_time, deadline, seed=42):
    """
    Simulated Annealing with periodic local polishing.
    """
    n = points.shape[0]
    rng = np.random.default_rng(seed)
    
    current = points.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    # SA Parameters
    T_start = 0.1
    T_end = 0.001
    n_iter = 2000 # Fixed iterations, time-checked inside
    cooling_rate = (T_end / T_start) ** (1.0 / n_iter)
    
    T = T_start
    polish_counter = 0
    last_polish_time = start_time
    
    for i in range(n_iter):
        if time.time() > deadline - 5:
            break
        
        # Adaptive step size: larger when T is high
        step_size = 0.05 * (1 + T)
        
        # Perturb random subset of points to allow larger moves
        # Sometimes move all, sometimes move few
        if rng.random() < 0.2:
            # Move all points slightly
            noise = rng.standard_normal((n, 2)) * step_size
            candidate = current + noise
        else:
            # Move 1-3 points more significantly
            n_move = rng.randint(1, 4)
            idx = rng.choice(n, size=n_move, replace=False)
            candidate = current.copy()
            for idx_i in idx:
                candidate[idx_i] += rng.standard_normal(2) * step_size * 2
        
        candidate_score = _compute_score(candidate)
        
        # Acceptance
        if candidate_score >= current_score:
            current = candidate
            current_score = candidate_score
            if current_score > best_score:
                best = candidate.copy()
                best_score = current_score
                # Polish the new best immediately if it's significantly better
                if best_score > 0.1: # Only polish if score is reasonable
                    if time.time() > last_polish_time + 5: # Don't polish too often
                        polished = _local_refine_powell(best, max_iter=50)
                        polished_score = _compute_score(polished)
                        if polished_score > best_score:
                            best = polished
                            best_score = polished_score
                            current = polished
                            current_score = polished_score
                        last_polish_time = time.time()
        else:
            # Metropolis criterion
            # Using log difference for stability
            delta_log = np.log(max(current_score, 1e-10)) - np.log(max(candidate_score, 1e-10))
            if delta_log < 0:
                delta_log = 0
            prob = np.exp(-delta_log / T)
            if rng.random() < prob:
                current = candidate
                current_score = candidate_score
        
        T *= cooling_rate
        
        # Periodic polish of the current state if stuck
        if i % 500 == 0 and i > 0:
            if time.time() > last_polish_time + 10:
                polished = _local_refine_powell(current, max_iter=50)
                polished_score = _compute_score(polished)
                if polished_score > current_score:
                    current = polished
                    current_score = polished_score
                    if current_score > best_score:
                        best = current.copy()
                        best_score = current_score
                last_polish_time = time.time()

    return best, best_score

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    start_time = time.time()
    deadline = start_time + 590 # Safety margin
    
    configs = _generate_initial_configs(n)
    
    best_points = configs[0]
    best_score = -1.0
    
    # Phase 1: Local refinement of initial configs
    for config in configs:
        if time.time() > deadline - 60:
            break
        refined = _local_refine_powell(config, max_iter=100)
        score = _compute_score(refined)
        if score > best_score:
            best_score = score
            best_points = refined.copy()
            
    # Phase 2: Simulated Annealing from the best config
    if time.time() < deadline - 30:
        sa_points, sa_score = _simulated_annealing(best_points, start_time, deadline, seed=1)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()
            
        # Final polish
        final_refined = _local_refine_powell(best_points, max_iter=200)
        final_score = _compute_score(final_refined)
        if final_score > best_score:
            best_score = final_score
            best_points = final_refined.copy()
            
    # Phase 3: Try SA from other good configs if time permits
    # Re-evaluate initial configs to find the 2nd best
    scored_configs = []
    for config in configs:
        # Quick score
        score = _compute_score(config)
        scored_configs.append((score, config))
    scored_configs.sort(key=lambda x: -x[0])
    
    # Skip the first one as we already used it
    for score, config in scored_configs[1:4]:
        if time.time() > deadline - 30:
            break
        
        # Refine this config
        refined = _local_refine_powell(config, max_iter=100)
        ref_score = _compute_score(refined)
        
        if ref_score > best_score:
            best_score = ref_score
            best_points = refined.copy()
            continue
            
        # Run SA from this refined config
        sa_points, sa_score = _simulated_annealing(refined, start_time, deadline, seed=100)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()
            
        # Polish
        final_refined = _local_refine_powell(best_points, max_iter=100)
        final_score = _compute_score(final_refined)
        if final_score > best_score:
            best_score = final_score
            best_points = final_refined.copy()

    # Ensure valid output
    if best_points.shape != (n, 2):
        best_points = configs[0]
        
    return best_points

# EVOLVE-BLOCK-END
