"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Finds a packing of 50 spheres in a unit cube [-0.5, 0.5]^3.
    Uses Simulated Annealing to directly maximize the minimum radius.
    """
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer for safety

    def compute_min_radius(X):
        """
        Computes the minimum radius r for a given configuration X.
        r = min( min_{i<j} ||x_i - x_j|| / 2, min_i min_k (0.5 - |x_ik|) )
        """
        # Pairwise distances
        # Vectorized calculation
        diffs = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dists_sq = np.sum(diffs ** 2, axis=2)
        # Mask diagonal to infinity
        np.fill_diagonal(dists_sq, np.inf)
        min_pair_dist = np.sqrt(np.min(dists_sq)) / 2.0
        
        # Wall distances
        wall_dists = 0.5 - np.abs(X)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist, min_wall_dist)

    def run_sa_direct(num_steps, seed=42, X_init=None):
        """
        Run Simulated Annealing to maximize the minimum radius.
        """
        rng = np.random.default_rng(seed)
        
        if X_init is None:
            # Initialize randomly
            X = rng.uniform(-0.5, 0.5, size=(N, 3))
        else:
            X = X_init.copy()
            
        r_current = compute_min_radius(X)
        best_r = r_current
        best_X = X.copy()
        
        T_start = 0.05
        T_end = 1e-5
        cooling_rate = (T_end / T_start) ** (1.0 / num_steps)
        T = T_start
        
        noise_scale = 0.05
        
        no_improve_count = 0
        max_no_improve = 2000
        
        for step in range(num_steps):
            if time.time() - start_time > time_limit:
                break
            
            # Pick random index
            i = rng.integers(0, N)
            
            # Old position
            pos_old = X[i].copy()
            
            # New position: Gaussian perturbation
            pos_new = pos_old + rng.normal(0, noise_scale, size=3)
            
            # Clip to cube
            pos_new = np.clip(pos_new, -0.5, 0.5)
            
            # Calculate new radius
            X[i] = pos_new
            r_new = compute_min_radius(X)
            
            delta_r = r_new - r_current
            
            # Acceptance criterion:
            # Accept if improvement.
            # Accept if worsening with probability exp(delta_r / T)
            if delta_r >= 0:
                accept = True
            else:
                if T > 1e-10:
                    accept = rng.random() < np.exp(delta_r / T)
                else:
                    accept = False
            
            if accept:
                r_current = r_new
                if r_current > best_r:
                    best_r = r_current
                    best_X = X.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Reject move
                X[i] = pos_old
                no_improve_count += 1
                
            T *= cooling_rate
            
            # Restart logic: if stuck, restart with perturbed best_X
            if no_improve_count > max_no_improve:
                # Perturb best_X
                X = best_X + rng.normal(0, 0.1, size=(N, 3))
                X = np.clip(X, -0.5, 0.5)
                r_current = compute_min_radius(X)
                no_improve_count = 0
                # Reset temperature slightly
                T = max(T, 0.01)

        return best_r, best_X

    # Run multiple SA instances with different seeds
    # We can run one long instance or several shorter ones.
    # Given N=50, compute_min_radius is fast.
    # Let's try a few runs.
    
    best_overall_r = 0.0
    best_overall_X = None
    
    seeds = [42, 123, 456, 789, 1011]
    
    # Estimate steps per run based on time
    # Let's allocate time evenly
    num_seeds = len(seeds)
    time_per_run = (time_limit - (time.time() - start_time)) / num_seeds
    # Assume ~1ms per step? No, let's just fix steps and check time.
    # 50k steps per run should be fast enough.
    
    for idx, seed in enumerate(seeds):
        if time.time() - start_time > time_limit - 2.0:
            break
            
        # If we have a previous best, use it as initial guess for subsequent runs?
        # Or just random start.
        # Using best_X as start for next run can help local refinement.
        X_init = best_overall_X if best_overall_X is not None else None
        
        r, X = run_sa_direct(num_steps=40000, seed=seed, X_init=X_init)
        
        if r > best_overall_r:
            best_overall_r = r
            best_overall_X = X
            
    if best_overall_X is None:
        # Fallback: Grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_overall_X = grid[:N]
        
    return best_overall_X

# EVOLVE-BLOCK-END
