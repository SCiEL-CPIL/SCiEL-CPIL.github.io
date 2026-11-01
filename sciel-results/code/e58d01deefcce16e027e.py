# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Place 50 equal circles in a unit circle to maximize the common radius.
    Multi-start Nelder-Mead with soft-min objective and local max-min polish.
    """
    
    def get_radius(x):
        pts = x.reshape(N, 2)
        d_origin = np.linalg.norm(pts, axis=1)
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        min_pair_dist = np.sqrt(np.min(d2))
        min_boundary_dist = 1.0 - np.max(d_origin)
        return min(min_pair_dist / 2.0, min_boundary_dist)

    def objective(x):
        """
        Negative of a smooth approximation to the minimum radius.
        """
        pts = x.reshape(N, 2)
        d_origin = np.linalg.norm(pts, axis=1)
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        
        d_pairs = np.sqrt(d2) / 2.0
        d_boundary = 1.0 - d_origin
        
        all_dists = np.concatenate([d_pairs.flatten(), d_boundary])
        all_dists = np.maximum(all_dists, 1e-12)
        
        alpha = 100.0
        scaled = -alpha * all_dists
        max_scaled = np.max(scaled)
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -log_sum / alpha
        
        return -soft_min

    def local_search_polish(x, iterations=500):
        """
        Local search to maximize the minimum distance.
        Identifies the bottleneck (closest pair or boundary violation) and resolves it.
        """
        pts = x.reshape(N, 2).copy()
        
        for _ in range(iterations):
            # Calculate distances
            diff = pts[:, None, :] - pts[None, :, :]
            d2 = np.sum(diff ** 2, axis=2)
            np.fill_diagonal(d2, np.inf)
            d = np.sqrt(d2)
            
            d_origin = np.linalg.norm(pts, axis=1)
            d_boundary = 1.0 - d_origin
            
            r_pair = np.min(d) / 2.0
            r_bound = np.min(d_boundary)
            
            current_r = min(r_pair, r_bound)
            
            if r_pair < r_bound:
                # Bottleneck is a pair
                idx = np.unravel_index(np.argmin(d), d.shape)
                i, j = idx
                if i == j: continue
                
                vec = pts[i] - pts[j]
                dist = np.linalg.norm(vec)
                if dist < 1e-10:
                    # Random nudge
                    pts[i] += np.random.normal(0, 0.001, 2)
                    pts[j] -= np.random.normal(0, 0.001, 2)
                else:
                    # Push apart to increase distance to 2*current_r + small epsilon
                    # Actually, we just push them apart by a small fraction of the vector
                    # to allow them to separate.
                    unit_vec = vec / dist
                    # We want to increase d.
                    # Step size proportional to distance?
                    step = unit_vec * 0.01
                    pts[i] += step
                    pts[j] -= step
            else:
                # Bottleneck is boundary
                idx = np.argmin(d_boundary)
                k = idx
                vec = pts[k]
                dist = np.linalg.norm(vec)
                if dist < 1e-10:
                    pts[k] += np.random.normal(0, 0.001, 2)
                else:
                    # Push inward
                    unit_vec = vec / dist
                    step = unit_vec * 0.01
                    pts[k] -= step
            
            # Project back to unit disk if necessary
            d_origin = np.linalg.norm(pts, axis=1)
            too_far = d_origin > 1.0
            if np.any(too_far):
                pts[too_far] *= (1.0 / d_origin[too_far])[:, None]
            
            # Optional: Check if we improved
            # new_r = get_radius(pts)
            # if new_r <= current_r - 1e-9:
            #     break # Stalled
            
        return pts.flatten()

    best_x = None
    best_r = -1.0
    n_starts = 30
    
    for i in range(n_starts):
        # Random initialization in unit disk
        rng = np.random.default_rng(i)
        angles = rng.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(rng.uniform(0, 1, N)) * 0.95
        x0 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).flatten()
        
        try:
            # Phase 1: Coarse Optimization
            res = minimize(objective, x0, method='Nelder-Mead',
                           options={'maxiter': 3000, 'maxfev': 3000, 'adaptive': True, 'xatol': 1e-6, 'fatol': 1e-6})
            
            x_opt = res.x
            r_opt = get_radius(x_opt)
            
            # Phase 2: Polish
            x_polished = local_search_polish(x_opt, iterations=100)
            r_polished = get_radius(x_polished)
            
            # Phase 3: Fine Optimization on Polished Result
            if r_polished > best_r:
                # Re-optimize around the polished point with tighter tolerance
                res2 = minimize(objective, x_polished, method='Nelder-Mead',
                                options={'maxiter': 2000, 'maxfev': 2000, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-8})
                x_final = res2.x
                r_final = get_radius(x_final)
                
                # Final Polish
                x_final_polished = local_search_polish(x_final, iterations=100)
                r_final_polished = get_radius(x_final_polished)
                
                if r_final_polished > best_r:
                    best_r = r_final_polished
                    best_x = x_final_polished
        except Exception:
            pass

    if best_x is None:
        return np.random.rand(N, 2) * 2 - 1

    return best_x.reshape(N, 2)
# EVOLVE-BLOCK-END
