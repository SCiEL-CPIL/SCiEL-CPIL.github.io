"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(centers):
    """
    Compute the packing radius: min(min_pair_dist/2, min_wall_dist).
    Returns the radius.
    """
    # Wall distances
    min_wall = np.min(0.5 - np.abs(centers))
    
    # Pairwise distances using broadcasting
    diff = centers[:, np.newaxis] - centers[np.newaxis, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_pair / 2.0, min_wall)


def construct():
    """
    Use simulated annealing to maximize the packing radius.
    """
    rng = np.random.default_rng(42)
    
    # Time limit check
    start_time = time.time()
    max_time = 115.0  # Leave 5s margin for safety
    
    best_centers = None
    best_score = -np.inf
    
    # Multiple restarts with different initializations
    n_restarts = 5
    
    for restart in range(n_restarts):
        # Check time
        if time.time() - start_time > max_time - 20.0:
            break
        
        # Initialization: Grid with jitter
        # 7x8 grid covers 56 spots, we use 50
        rows, cols = 7, 8
        x_ticks = np.linspace(-0.45, 0.45, cols)
        y_ticks = np.linspace(-0.45, 0.45, rows)
        
        centers = np.zeros((N, 2))
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx < N:
                    centers[idx] = [x_ticks[c], y_ticks[r]]
                    idx += 1
                else:
                    break
            if idx >= N:
                break
        
        # Add jitter to break symmetry
        jitter_scale = 0.02 + 0.01 * restart
        centers += rng.normal(0, jitter_scale, (N, 2))
        centers = np.clip(centers, -0.4999, 0.4999)
        
        # Precompute initial score
        current_score = _score(centers)
        
        best_centers_local = centers.copy()
        best_score_local = current_score
        
        # Simulated annealing parameters
        T_start = 0.03
        T_end = 1e-10
        n_iterations = 500000
        alpha = (T_end / T_start) ** (1.0 / n_iterations)
        
        base_step = 0.04
        step_size = base_step
        
        no_improve_count = 0
        max_no_improve = 5000
        
        for it in range(n_iterations):
            # Check time
            if it % 5000 == 0:
                if time.time() - start_time > max_time - 5.0:
                    break
            
            T = T_start * (alpha ** it)
            
            # Pick a random center
            i = rng.integers(N)
            old_pos = centers[i].copy()
            
            # Perturb
            centers[i] += rng.normal(0, step_size, 2)
            centers[i] = np.clip(centers[i], -0.4999, 0.4999)
            
            # Compute new score
            new_score = _score(centers)
            
            if new_score >= current_score:
                current_score = new_score
                no_improve_count = 0
                if new_score > best_score_local:
                    best_score_local = new_score
                    best_centers_local = centers.copy()
            else:
                delta = new_score - current_score
                # Accept with prob exp(delta/T)
                if delta / T > -100:
                    if rng.random() < np.exp(delta / T):
                        current_score = new_score
                        no_improve_count = 0
                    else:
                        centers[i] = old_pos
                        no_improve_count += 1
                else:
                    centers[i] = old_pos
                    no_improve_count += 1
            
            # Adaptive step size
            if no_improve_count > max_no_improve:
                step_size *= 0.5
                no_improve_count = 0
            elif it % 2000 == 0 and it > 0:
                step_size *= 0.99
                
            # Periodically restart step size to escape local minima
            if it % 30000 == 0 and it > 0:
                step_size = base_step * (0.5 + 0.5 * rng.random())
                no_improve_count = 0
        
        # Update global best
        if best_score_local > best_score:
            best_score = best_score_local
            best_centers = best_centers_local
    
    return best_centers if best_centers is not None else centers
# EVOLVE-BLOCK-END
