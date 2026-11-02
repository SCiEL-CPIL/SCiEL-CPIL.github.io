"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _compute_energy(pts):
    """Compute logarithmic energy using math.fsum for precision."""
    n = len(pts)
    dots = np.dot(pts, pts.T)
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    dist = np.sqrt(dist_sq)
    log_inv_dist = -np.log(dist)
    np.fill_diagonal(log_inv_dist, 0.0)
    triu_indices = np.triu_indices(n, k=1)
    energy = math.fsum(log_inv_dist[triu_indices].tolist())
    return energy

def _project_to_sphere(pts):
    """Normalize each point to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _sa_run(start_pts, T_start=0.5, alpha=0.995, max_iters=5000, seed=42):
    """
    Run Simulated Annealing.
    """
    rng = np.random.default_rng(seed)
    pts = _project_to_sphere(start_pts.copy())
    current_energy = _compute_energy(pts)
    best_energy = current_energy
    best_pts = pts.copy()
    
    for i in range(max_iters):
        T = T_start * (alpha ** i)
        if T < 1e-8:
            break
            
        # Generate move
        # Adaptive step size based on T? Or fixed?
        # Let's use a mix of local and global moves.
        move_type = rng.random()
        
        old_pts = pts.copy()
        
        if move_type < 0.7:
            # Local perturbation: move one point slightly
            idx = rng.integers(N)
            # Step size scales with T to allow larger jumps at high T
            step_size = 0.1 * math.sqrt(T / T_start + 0.1) 
            noise = rng.normal(0, step_size, 3)
            pts[idx] += noise
            pts[idx] /= np.linalg.norm(pts[idx])
        elif move_type < 0.9:
            # Global move: replace one point with random
            idx = rng.integers(N)
            pts[idx] = rng.normal(size=3)
            pts[idx] /= np.linalg.norm(pts[idx])
        else:
            # Swap two points
            i1, i2 = rng.choice(N, 2, replace=False)
            pts[i1], pts[i2] = pts[i2].copy(), pts[i1].copy()
            
        new_energy = _compute_energy(pts)
        
        if not np.isfinite(new_energy):
            pts = old_pts
            continue
            
        delta = new_energy - current_energy
        
        # Metropolis criterion
        if delta < 0 or rng.random() < math.exp(-delta / T):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        else:
            pts = old_pts
            
    return best_pts, best_energy

def construct():
    """Construct 150 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    time_limit = 115.0  # Leave some buffer
    
    best_pts_overall = None
    best_energy_overall = float('inf')
    
    # Initial configurations
    configs = []
    
    # 1. Fibonacci spiral (good starting point)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    configs.append(fib_pts)
    
    # 2. Random starts
    rng_init = np.random.default_rng(0)
    for _ in range(3):
        rand_pts = rng_init.normal(size=(N, 3))
        configs.append(_project_to_sphere(rand_pts))
        
    # Run SA on each config
    for idx, start_pts in enumerate(configs):
        if time.time() - start_time > time_limit:
            break
            
        # Use different seeds for each run
        seed = 100 + idx
        
        try:
            optimized_pts, energy = _sa_run(start_pts, T_start=0.5, alpha=0.995, max_iters=8000, seed=seed)
            if energy < best_energy_overall:
                best_energy_overall = energy
                best_pts_overall = optimized_pts
        except Exception:
            pass

    # If time remains, do a final polishing run with a very low T (essentially hill climbing)
    if best_pts_overall is not None and time.time() - start_time < time_limit - 10:
        try:
            # Use a very low start T to act as a local optimizer
            polished_pts, energy = _sa_run(best_pts_overall, T_start=0.01, alpha=0.99, max_iters=5000, seed=999)
            if energy < best_energy_overall:
                best_energy_overall = energy
                best_pts_overall = polished_pts
        except Exception:
            pass

    if best_pts_overall is None:
        return _project_to_sphere(fib_pts)
        
    return _project_to_sphere(best_pts_overall)

# EVOLVE-BLOCK-END
