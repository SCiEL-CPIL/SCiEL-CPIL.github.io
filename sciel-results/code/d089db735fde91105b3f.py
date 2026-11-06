"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    # Start with the regular octagon
    n = 8
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Regular octagon with radius 1 (diameter 2)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)]).flatten()
    
    # We want to maximize the product of distances after scaling to diameter 2.
    # Score is proportional to product of distances / (Diameter/2)^(n(n-1))
    # Let's maximize log(product of distances) - n(n-1) * log(Diameter/2)
    # This is equivalent to maximizing sum(log(d_ij)) - n(n-1) * log(D) + const
    
    def objective(x):
        pts = x.reshape(n, 2)
        # Check for non-finite values
        if not np.all(np.isfinite(pts)):
            return 1e10
            
        # Calculate pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        dists_masked = dists[mask]
        
        if len(dists_masked) == 0:
            return 1e10
            
        # Check for zero distances (coincident points)
        if np.any(dists_masked < 1e-10):
            return 1e10
            
        d_max = np.max(dists_masked)
        
        if d_max < 1e-10:
            return 1e10
            
        # Log product of all ordered pairs is 2 * sum(log(unordered pairs))
        # We maximize log(Delta) where Delta is product of scaled distances.
        # Scaled distance d'_ij = d_ij / (d_max / 2)
        # log(Delta) = sum_{i != j} log(d'_ij) = sum_{i != j} (log(d_ij) - log(d_max/2))
        # = 2 * sum_{i < j} log(d_ij) - n(n-1) * log(d_max/2)
        
        sum_log_dists = np.sum(np.log(dists_masked))
        # dists_masked contains unordered pairs? No, dists is full matrix, mask removes diagonal.
        # So dists_masked has n*(n-1) elements (ordered pairs).
        # So sum_log_dists is already sum over ordered pairs.
        
        log_scaled_product = sum_log_dists - n * (n - 1) * np.log(d_max / 2.0)
        
        # We minimize negative
        return -log_scaled_product

    start_time = time.time()
    timeout = 140.0
    
    best_x = x0
    best_val = objective(x0)
    
    # Try a few different initializations or just refine the regular octagon
    # The regular octagon is a symmetric local optimum. We might need to break symmetry or try different topologies.
    # However, for n=8, the regular octagon is often the global max for "max product of distances" if constrained to circle?
    # Without circle constraint, maybe a "two squares" configuration (square of radius r1 and square of radius r2 rotated 45 deg) is better?
    
    # Let's try optimizing from the regular octagon first.
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
        
        if time.time() - start_time < timeout:
            if res.success or res.fun < best_val:
                if np.all(np.isfinite(res.x)):
                    best_x = res.x
                    best_val = res.fun
                    
            # Try a second run with slightly perturbed start to escape local minima
            if time.time() - start_time < timeout:
                # Perturb angles slightly
                angles_pert = angles + np.random.uniform(-0.1, 0.1, n)
                x_pert = np.column_stack([np.cos(angles_pert), np.sin(angles_pert)]).flatten()
                # Also try scaling
                x_pert = x_pert * 1.1
                
                res2 = minimize(objective, x_pert, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
                
                if np.all(np.isfinite(res2.x)) and res2.fun < best_val:
                    best_x = res2.x
                    best_val = res2.fun
                    
    except Exception:
        pass

    # Final sanity check
    best_pts = best_x.reshape(n, 2)
    if not np.all(np.isfinite(best_pts)):
        # Fallback to regular octagon
        best_pts = x0.reshape(n, 2)
        
    # Ensure coordinates are moderate
    max_coord = np.max(np.abs(best_pts))
    if max_coord > 1e6 or max_coord < 1e-6:
        # Rescale to unit circle
        norms = np.linalg.norm(best_pts, axis=1)
        norms[norms == 0] = 1.0
        best_pts = best_pts / norms[:, np.newaxis]

    return best_pts.tolist()
# EVOLVE-BLOCK-END
