# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time
from itertools import combinations

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    
    def get_score(pts):
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(x):
        pts = x.reshape(n, 2)
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_min <= 1e-9:
            return 1e10
        # Minimize -log(ratio)
        return -np.log(d_min / d_max)

    def hex_inits():
        inits = []
        pts = []
        for j in range(-4, 5):
            for i in range(-4, 5):
                x = i + 0.5 * j
                y = (np.sqrt(3) / 2) * j
                pts.append([x, y])
        pts = np.array(pts)
        dists_to_origin = np.linalg.norm(pts, axis=1)
        
        idx_sorted = np.argsort(dists_to_origin)
        
        # The 13 closest points form the core (1 center + 6 ring1 + 6 ring2_inner)
        core_indices = idx_sorted[:13]
        
        # Find the next ring (6 points)
        # The 14th point (index 13) is the first of the outer ring
        dist_outer_val = dists_to_origin[idx_sorted[13]]
        
        outer_candidates_indices = []
        for idx in idx_sorted[13:]:
            if np.isclose(dists_to_origin[idx], dist_outer_val, atol=1e-6):
                outer_candidates_indices.append(idx)
            else:
                break
        
        # If we didn't find exactly 6, just take whatever is there
        # But logically there should be 6
        if len(outer_candidates_indices) < 3:
             # Fallback: just take the next 3 available
             selected_indices = list(core_indices) + list(idx_sorted[13:16])
             inits.append(pts[selected_indices])
             return inits

        for comb in combinations(outer_candidates_indices, 3):
            selected_indices = list(core_indices) + list(comb)
            init_pts = pts[selected_indices]
            inits.append(init_pts)
            
        return inits

    def circle_init():
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def random_init(seed):
        np.random.seed(seed)
        return np.random.randn(n, 2) * 0.5

    inits = hex_inits()
    inits.append(circle_init())
    for i in range(5):
        inits.append(random_init(i + 100))
    
    best_score = -1
    best_pts = inits[0]
    
    start_time = time.time()
    timeout = 590 # Safety margin
    
    for i, init_pts in enumerate(inits):
        if time.time() - start_time > timeout:
            break
            
        # Normalize initial points to unit max distance to help convergence
        d_max_init = np.max(pdist(init_pts))
        if d_max_init > 1e-9:
            init_pts = init_pts / d_max_init
        else:
            # If degenerate, perturb
            init_pts = init_pts + np.random.randn(n, 2) * 0.01
            d_max_init = np.max(pdist(init_pts))
            init_pts = init_pts / d_max_init

        try:
            res = minimize(
                objective, 
                init_pts.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            res_pts = res.x.reshape(n, 2)
            res_score = get_score(res_pts)
            
            if res_score > best_score:
                best_score = res_score
                best_pts = res_pts
        except Exception:
            pass

    return best_pts
# EVOLVE-BLOCK-END
