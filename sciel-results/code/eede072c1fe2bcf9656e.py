"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
from scipy.spatial.distance import cdist

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time

def construct():
    """
    Place 50 equal spheres in unit cube [-0.5, 0.5]^3 to maximize minimum distance.
    Strategy:
    1. Generate a high-quality initial guess based on a 4x4x4 grid (64 points), 
       selecting 50 points.
    2. Refine using L-BFGS-B with a smoothed log-sum-exp objective.
    3. Polish using Nelder-Mead on the exact non-smooth objective.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin below 150s kill
    margin = 5.0
    effective_limit = time_limit - margin
    
    def check_time():
        return (time.time() - start_time) >= effective_limit
    
    def score_centres(centres):
        """Compute the radius (min of pairwise/2 and boundary distances)."""
        # Boundary distances
        # Distance to each face is 0.5 - |coord|
        # min_i min_k (0.5 - |c_ik|)
        boundary_dists = 0.5 - np.abs(centres)
        min_boundary = np.min(boundary_dists)
        
        # Pairwise distances
        if N > 1:
            dists = cdist(centres, centres)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists)
            min_pair_half = min_pair / 2.0
        else:
            min_pair_half = np.inf
        
        return min(min_boundary, min_pair_half)
    
    def soft_min_score(centres_flat, beta=100.0):
        """
        Smooth approximation of the negative radius.
        Objective to minimize: -soft_min(r_i)
        where r_i are the constraints (pairwise/2, boundary).
        """
        centres = centres_flat.reshape(N, 3)
        
        # Boundary terms: 0.5 - |c_ik|
        boundary_terms = 0.5 - np.abs(centres)  # (N, 3)
        
        # Pairwise distance terms: |c_i - c_j| / 2
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, np.inf)
        iu = np.triu_indices(N, k=1)
        pair_terms = dists[iu] / 2.0
        
        all_terms = np.concatenate([boundary_terms.flatten(), pair_terms])
        
        # Soft-min: -1/beta * log(sum(exp(-beta * x)))
        # To be stable, shift by max
        max_term = np.max(all_terms)
        exp_terms = np.exp(-beta * (all_terms - max_term))
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0:
            return 1e10
        soft_min_val = max_term - np.log(sum_exp) / beta
        
        return -soft_min_val

    def neg_score(centres_flat):
        """Negative exact score for direct methods."""
        centres = centres_flat.reshape(N, 3)
        if not np.all(np.isfinite(centres)):
            return 1e10
        return -score_centres(centres)

    # 1. Initial Guess Generation
    # 4x4x4 grid gives 64 points.
    # Coordinates: -0.375, -0.125, 0.125, 0.375
    # Spacing: 0.25. Boundary dist: 0.125. Pairwise dist: 0.25.
    # Radius = 0.125.
    # We need 50 points.
    
    # Generate 4x4x4 grid
    coords = np.array([-0.375, -0.125, 0.125, 0.375])
    grid = np.array([(x, y, z) for z in coords for y in coords for x in coords])
    
    # Select 50 points. 
    # To maintain symmetry, we should remove points from the corners of the 4x4x4 block?
    # Or just the first 50?
    # The 4x4x4 grid is highly symmetric. Removing any 14 points breaks symmetry.
    # Let's try removing points with the smallest "distance to center"? No, we want to keep space.
    # Actually, for packing, we want points to be as far apart as possible.
    # The 4x4x4 grid is a lattice. 
    # Let's just take the first 50. Or better, random 50 with a seed?
    # Deterministic is better for reproducibility.
    # Let's sort by index and take first 50.
    
    initial_centres = grid[:50].copy()
    best_centres = initial_centres
    best_score = score_centres(best_centres)
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # 2. Optimization Stage 1: Smooth L-BFGS-B
    # Start with moderate beta to find basin
    try:
        betas = [10.0, 50.0, 100.0, 200.0]
        current_centres = best_centres.copy()
        
        for beta in betas:
            if check_time():
                break
            
            def smooth_obj(centres_flat):
                return soft_min_score(centres_flat, beta=beta)
            
            result = minimize(
                smooth_obj,
                current_centres.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < smooth_obj(current_centres.flatten()):
                new_centres = result.x.reshape(N, 3)
                new_score = score_centres(new_centres)
                if new_score >= best_score - 1e-10:
                    current_centres = new_centres
                    if new_score > best_score:
                        best_centres = new_centres.copy()
                        best_score = new_score
                        
    except Exception:
        pass

    if check_time():
        return best_centres

    # 3. Optimization Stage 2: High Beta Polish
    try:
        current_centres = best_centres.copy()
        for beta in [400.0, 1000.0]:
            if check_time():
                break
            def smooth_obj_high(centres_flat):
                return soft_min_score(centres_flat, beta=beta)
            
            result = minimize(
                smooth_obj_high,
                current_centres.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if result.fun < smooth_obj_high(current_centres.flatten()):
                new_centres = result.x.reshape(N, 3)
                new_score = score_centres(new_centres)
                if new_score > best_score:
                    best_centres = new_centres.copy()
                    best_score = new_score
                    
    except Exception:
        pass

    if check_time():
        return best_centres

    # 4. Optimization Stage 3: Nelder-Mead on Exact Objective
    # Nelder-Mead doesn't use gradients, handles non-smoothness better.
    # It can be slow, so limit iterations.
    try:
        result = minimize(
            neg_score,
            best_centres.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        
        new_centres = result.x.reshape(N, 3)
        new_score = score_centres(new_centres)
        
        # Ensure validity
        if np.all(np.isfinite(new_centres)) and np.all(new_centres >= -0.5) and np.all(new_centres <= 0.5):
             if new_score > best_score:
                 best_centres = new_centres.copy()
                 best_score = new_score
                 
    except Exception:
        pass

    # Final Validation
    final_score = score_centres(best_centres)
    if not np.isfinite(final_score):
        # Fallback to initial grid
        best_centres = grid[:50].copy()

    return best_centres

# EVOLVE-BLOCK-END
