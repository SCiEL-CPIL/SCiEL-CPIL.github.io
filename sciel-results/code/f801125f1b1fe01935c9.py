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
    time_limit = 140.0  # Safety margin below 150s kill
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
    # Use a Halton sequence (low-discrepancy) to generate well-distributed points.
    # This breaks the lattice symmetry of the grid and provides a better spread.
    
    def halton_sequence(n, base):
        """Generate n points in the Halton sequence with given base."""
        result = []
        f = 1.0
        for i in range(1, n + 1):
            x = 0.0
            i_val = i
            while i_val > 0:
                f /= base
                x += f * (i_val % base)
                i_val //= base
            result.append(x)
        return np.array(result)
    
    # Generate 3D Halton sequence for 50 points
    # Bases must be coprime, e.g., 2, 3, 5
    h1 = halton_sequence(N, 2)
    h2 = halton_sequence(N, 3)
    h3 = halton_sequence(N, 5)
    
    # Map to [-0.5, 0.5]
    # We scale slightly inside to ensure we start with valid boundary distances
    # Halton points are in [0, 1]
    # Map to [-0.45, 0.45] to have some initial boundary clearance (0.05)
    # This avoids starting with points on the boundary which might cause issues with gradient steps
    scale = 0.9  # 0.9 * 0.5 = 0.45
    initial_centres = np.column_stack((
        h1 * scale - scale/2,
        h2 * scale - scale/2,
        h3 * scale - scale/2
    ))
    
    # Add small deterministic noise to break any accidental symmetries
    rng = np.random.default_rng(42)
    initial_centres += rng.normal(0, 0.01, size=(N, 3))
    
    # Clip to valid range just in case
    initial_centres = np.clip(initial_centres, -0.5, 0.5)
    
    best_centres = initial_centres.copy()
    best_score = score_centres(best_centres)
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # 2. Optimization Stage 1: Smooth L-BFGS-B
    # Start with moderate beta to find basin
    try:
        # Adjusted betas: start lower to allow large moves, then increase
        betas = [5.0, 20.0, 50.0, 100.0]
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
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
            )
            
            # Check if the result is valid and better
            if result.fun < smooth_obj(current_centres.flatten()):
                new_centres = result.x.reshape(N, 3)
                # Ensure validity
                if np.all(np.isfinite(new_centres)) and np.all(new_centres >= -0.5) and np.all(new_centres <= 0.5):
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
        # More aggressive high-beta polishing
        for beta in [300.0, 800.0, 2000.0]:
            if check_time():
                break
            def smooth_obj_high(centres_flat):
                return soft_min_score(centres_flat, beta=beta)
            
            result = minimize(
                smooth_obj_high,
                current_centres.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20}
            )
            
            if result.fun < smooth_obj_high(current_centres.flatten()):
                new_centres = result.x.reshape(N, 3)
                if np.all(np.isfinite(new_centres)) and np.all(new_centres >= -0.5) and np.all(new_centres <= 0.5):
                    new_score = score_centres(new_centres)
                    if new_score > best_score:
                        best_centres = new_centres.copy()
                        best_score = new_score
                        current_centres = new_centres # Continue from the new best
                    
    except Exception:
        pass

    if check_time():
        return best_centres

    # 4. Optimization Stage 3: Nelder-Mead on Exact Objective
    # Nelder-Mead doesn't use gradients, handles non-smoothness better.
    # It can be slow, so limit iterations.
    try:
        # Use a smaller initial simplex step size to avoid jumping out of the basin
        # Default is based on 0.05 * x + 0.00025. For x~0.5, step ~0.025.
        # We want to polish, so maybe smaller.
        result = minimize(
            neg_score,
            best_centres.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'initial_simplex': None, 'adaptive': True}
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
