"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time

def construct():
    """
    Place 50 equal spheres in unit cube [-0.5, 0.5]^3 to maximize minimum distance.
    Uses:
    1. Differential evolution for global exploration.
    2. Smooth soft-min surrogate with L-BFGS-B for gradient-based refinement.
    3. Final high-precision polish.
    """
    start_time = time.time()
    time_limit = 120.0
    margin = 10.0
    effective_limit = time_limit - margin
    
    def check_time():
        return (time.time() - start_time) >= effective_limit
    
    def score_centres(centres):
        """Compute the radius (min of pairwise/2 and boundary distances)."""
        boundary_dists = 0.5 - np.abs(centres)
        min_boundary = np.min(boundary_dists)
        
        if N > 1:
            dists = cdist(centres, centres)
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists)
        else:
            min_pair = np.inf
        
        return min(min_boundary, min_pair / 2.0)
    
    def soft_min_score(centres, beta=50.0):
        """
        Smooth approximation of the radius using log-sum-exp soft-min.
        Returns -soft_min to be minimized.
        """
        centres = centres.reshape(N, 3)
        
        # Boundary terms: 0.5 - |c_ik| for each coordinate
        boundary_terms = 0.5 - np.abs(centres)  # (N, 3)
        
        # Pairwise distance terms: |c_i - c_j| / 2 for i < j
        if N > 1:
            dists = cdist(centres, centres)
            np.fill_diagonal(dists, np.inf)
            # Take upper triangle
            iu = np.triu_indices(N, k=1)
            pair_terms = dists[iu] / 2.0
        else:
            pair_terms = np.array([np.inf])
        
        # Combine all terms
        all_terms = np.concatenate([boundary_terms.flatten(), pair_terms])
        
        # Soft-min via log-sum-exp: -log(sum(exp(-beta * x))) / beta
        # For numerical stability, subtract max
        max_term = np.max(all_terms)
        exp_terms = np.exp(-beta * (all_terms - max_term))
        soft_min_val = max_term - np.log(np.sum(exp_terms)) / beta
        
        return -soft_min_val
    
    def neg_score(centres_flat):
        """Negative exact score for minimization (used by DE)."""
        centres = centres_flat.reshape(N, 3)
        if not np.all(np.isfinite(centres)):
            return 1e10
        return -score_centres(centres)
    
    # Initial grid-based solution as fallback
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    best_centres = grid[:N].copy()
    best_score = score_centres(best_centres)
    
    # Stage 1: Differential Evolution
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        de_result = differential_evolution(
            neg_score, 
            bounds, 
            maxiter=25, 
            popsize=4, 
            seed=42, 
            tol=1e-5,
            polish=False
        )
        
        candidate = de_result.x.reshape(N, 3)
        cand_score = score_centres(candidate)
        if cand_score > best_score:
            best_centres = candidate
            best_score = cand_score
        
    except Exception as e:
        pass
    
    if check_time():
        return best_centres
    
    # Stage 2: Smooth surrogate optimization with L-BFGS-B
    # Use increasing beta for annealing
    try:
        current_centres = best_centres.copy()
        beta_schedule = [20.0, 50.0, 100.0, 200.0, 400.0]
        
        for beta in beta_schedule:
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
            
            if result.success or result.fun < smooth_obj(current_centres.flatten()):
                new_centres = result.x.reshape(N, 3)
                new_score = score_centres(new_centres)
                if new_score >= best_score - 1e-10:
                    current_centres = new_centres
                    if new_score > best_score:
                        best_centres = new_centres.copy()
                        best_score = new_score
            
            if check_time():
                break
        
        # Final polish with very high beta
        if not check_time():
            def high_beta_obj(centres_flat):
                return soft_min_score(centres_flat, beta=1000.0)
            
            result = minimize(
                high_beta_obj,
                current_centres.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if result.success:
                new_centres = result.x.reshape(N, 3)
                new_score = score_centres(new_centres)
                if new_score > best_score:
                    best_centres = new_centres.copy()
                    best_score = new_score
    
    except Exception as e:
        pass
    
    # Final check
    final_score = score_centres(best_centres)
    if not np.isfinite(final_score):
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centres = grid[:N]
    
    return best_centres

# EVOLVE-BLOCK-END
