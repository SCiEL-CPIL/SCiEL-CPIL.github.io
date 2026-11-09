import numpy as np
from scipy.optimize import minimize
import time
from scipy.spatial.distance import cdist

N = 50

def construct():
    """
    Place 50 equal spheres in unit cube [-0.5, 0.5]^3 to maximize minimum distance.
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin below 150s kill
    margin = 5.0
    effective_limit = time_limit - margin
    
    def check_time():
        return (time.time() - start_time) >= effective_limit

    def score_centres(centres):
        """Compute the radius (min of pairwise/2 and boundary distances)."""
        # Boundary distances: distance to each face is 0.5 - |coord|
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
        # If max_term is very negative, exp(-beta * (x - max_term)) might underflow/overflow
        # x - max_term <= 0. So exp(-beta * (x - max_term)) >= 1.
        # Actually, let's rewrite:
        # softmin = -1/beta * log(sum(exp(-beta * x)))
        # = -1/beta * log(exp(-beta * max_term) * sum(exp(-beta * (x - max_term))))
        # = max_term - 1/beta * log(sum(exp(-beta * (x - max_term))))
        
        exp_terms = np.exp(-beta * (all_terms - max_term))
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0 or not np.isfinite(sum_exp):
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
    # Use Halton sequence for low-discrepancy initial points
    def halton_sequence(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    def generate_initial_centres(n_points, seed=42):
        np.random.seed(seed)
        centres = np.zeros((n_points, 3))
        for i in range(n_points):
            for j in range(3):
                # Use different bases for each dimension to ensure low discrepancy
                # Bases 2, 3, 5 are coprime
                base = [2, 3, 5][j]
                centres[i, j] = halton_sequence(i + 1, base) - 0.5
        return centres

    initial_centres = generate_initial_centres(N)
    
    # Simple repulsion step to spread points out
    # This helps escape the "clumped" nature of raw Halton sequences near boundaries
    for _ in range(100):
        if check_time():
            break
        forces = np.zeros_like(initial_centres)
        for i in range(N):
            for j in range(i+1, N):
                diff = initial_centres[i] - initial_centres[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    dist = 1e-6
                # Repulsive force
                forces[i] += diff / (dist**3)
                forces[j] -= diff / (dist**3)
        # Boundary repulsion
        for i in range(N):
            for k in range(3):
                if initial_centres[i, k] > 0:
                    forces[i, k] -= 1.0 / (0.5 - initial_centres[i, k])**2
                else:
                    forces[i, k] += 1.0 / (0.5 + initial_centres[i, k])**2
        
        # Update positions with small step
        step_size = 0.01
        initial_centres += step_size * forces
        # Project back to cube
        initial_centres = np.clip(initial_centres, -0.49, 0.49)
        
    best_centres = initial_centres.copy()
    best_score = score_centres(best_centres)
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # 2. Optimization Stage 1: Smooth L-BFGS-B with Annealing
    # Start with moderate beta to find basin
    try:
        betas = [5.0, 20.0, 50.0, 100.0, 200.0]
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
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            new_centres = result.x.reshape(N, 3)
            new_score = score_centres(new_centres)
            
            # Accept if it's valid and better or equal in the smooth sense
            # We check exact score to track progress
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
        for beta in [400.0, 800.0]:
            if check_time():
                break
            def smooth_obj_high(centres_flat):
                return soft_min_score(centres_flat, beta=beta)
            
            result = minimize(
                smooth_obj_high,
                current_centres.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
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
    # Nelder-Mead can handle the non-smoothness of the exact min function.
    # It is slower per iteration but can refine the local minimum more accurately.
    try:
        # Use a small initial simplex size
        initial_simplex = best_centres.flatten()
        # Perturb slightly to ensure simplex is not degenerate
        simplex = [initial_simplex]
        for i in range(N*3):
            point = initial_simplex.copy()
            point[i] += 1e-4
            simplex.append(point)
            
        result = minimize(
            neg_score,
            initial_simplex,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
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
        # Fallback to initial repulsion result
        best_centres = initial_centres.copy()

    return best_centres

# EVOLVE-BLOCK-END
