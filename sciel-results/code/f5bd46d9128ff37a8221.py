import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave some margin for safety
    
    best_c = None
    best_r = 0.0
    
    # Helper to calculate the actual radius (min distance)
    def calc_radius(c):
        c = c.reshape(N, 3)
        # Pairwise distances
        # Use broadcasting
        # diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        # dists = np.linalg.norm(diff, axis=2)
        # np.fill_diagonal(dists, np.inf)
        # min_pair = np.min(dists) / 2.0
        
        # Wall distances
        # d_wall = 0.5 - np.max(np.abs(c), axis=1)
        # min_wall = np.min(d_wall)
        
        # Vectorized pairwise for speed
        # N is 50, so N^2 is 2500. Very fast.
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Mask diagonal
        idx = np.where(np.triu(np.ones(dists.shape, dtype=bool), k=1))
        min_pair = np.min(dists[idx]) / 2.0
        
        min_wall = 0.5 - np.max(np.abs(c), axis=1)
        min_wall_val = np.min(min_wall)
        
        return min(min_pair, min_wall_val)

    # Soft min objective to minimize
    # We want to maximize min(dists), so minimize -softmin(dists)
    def objective(c_flat, lam):
        c = c_flat.reshape(N, 3)
        # Clip to ensure we don't evaluate outside if bounds fail? 
        # Bounds should handle it.
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Get upper triangle indices
        iu = np.triu_indices(N, k=1)
        pair_dists = dists[iu]
        
        wall_dists = 0.5 - np.max(np.abs(c), axis=1)
        
        all_dists = np.concatenate([pair_dists, wall_dists])
        
        # Soft min: -1/lam * log(sum(exp(-lam * d)))
        # To avoid overflow, subtract max
        # Note: we want to minimize this value.
        # If d is large, exp(-lam*d) is small.
        # If d is small, exp(-lam*d) is large.
        # Sum is dominated by smallest d.
        # log(sum) approx log(exp(-lam*min_d)) = -lam*min_d.
        # -1/lam * (-lam*min_d) = min_d.
        # So softmin approx min_d.
        # We want to MAXIMIZE min_d.
        # So we MINIMIZE -softmin.
        
        # Shift for numerical stability
        m = np.min(all_dists)
        if m < 0:
            # If any distance is negative (overlap or out of bounds), penalty heavily
            return 1000.0 - m # Return a large positive value to minimize? 
            # Actually, if m < 0, the configuration is invalid.
            # We want to drive it back.
            # Let's just return a large number.
        
        vals = np.exp(-lam * (all_dists - m))
        soft_min_val = m - (1.0/lam) * np.log(np.sum(vals))
        
        return -soft_min_val

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    n_restarts = 20
    
    for i in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        # Ensure points are not too close initially to help convergence?
        # Simple random in [-0.4, 0.4] to stay inside
        c0 = np.random.uniform(-0.4, 0.4, (N, 3))
        
        # Try a few lambda values
        # Start with lower lambda for global-ish search, then higher
        lams = [5.0, 20.0, 100.0]
        
        current_c = c0.flatten()
        
        for lam in lams:
            if time.time() - start_time > time_limit:
                break
                
            try:
                res = minimize(
                    objective, 
                    current_c, 
                    args=(lam,), 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-8}
                )
                current_c = res.x
            except Exception:
                pass
        
        c_opt = current_c.reshape(N, 3)
        r_opt = calc_radius(c_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt

    if best_c is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]

    return best_c
