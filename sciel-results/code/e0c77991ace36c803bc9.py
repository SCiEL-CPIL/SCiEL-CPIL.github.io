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
    1. Generate multiple high-quality initial guesses (Halton, grid, random).
    2. Refine each using L-BFGS-B with a smoothed log-sum-exp objective to find a good basin.
    3. Polish the best result using a targeted local search that specifically 
       pushes apart the binding (closest) pair or moves boundary-violating points inward.
    """
    start_time = time.time()
    time_limit = 150.0
    safety_margin = 5.0
    effective_limit = time_limit - safety_margin
    
    def check_time():
        return (time.time() - start_time) >= effective_limit

    def score_centres(centres):
        """
        Compute the radius (min of pairwise/2 and boundary distances).
        Returns 0.0 if invalid.
        """
        if not np.all(np.isfinite(centres)):
            return 0.0
        if np.any(centres < -0.5) or np.any(centres > 0.5):
            return 0.0
        
        # Boundary distances: distance to nearest face
        boundary_dists = 0.5 - np.abs(centres)
        min_boundary = np.min(boundary_dists)
        
        # Pairwise distances
        if N > 1:
            dists = cdist(centres, centres, metric='euclidean')
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists)
            min_pair_half = min_pair / 2.0
        else:
            min_pair_half = np.inf
        
        return min(min_boundary, min_pair_half)

    def get_binding_info(centres):
        """
        Identify the binding constraint.
        Returns:
        - 'pair', i, j if the closest pair is binding.
        - 'boundary', i, k if the closest boundary point is binding.
        """
        # Boundary check
        boundary_dists = 0.5 - np.abs(centres)
        min_boundary_val = np.min(boundary_dists)
        min_boundary_idx = np.unravel_index(np.argmin(boundary_dists), boundary_dists.shape)
        
        # Pair check
        dists = cdist(centres, centres, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        min_pair_val = np.min(dists) / 2.0
        min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
        
        if min_boundary_val < min_pair_val:
            return 'boundary', min_boundary_idx[0], min_boundary_idx[1], min_boundary_val
        else:
            return 'pair', min_pair_idx[0], min_pair_idx[1], min_pair_val

    def local_search(centres, max_iters=500):
        """
        Targeted local search.
        Iteratively finds the binding constraint and moves the involved points 
        to increase the minimum distance.
        """
        current_centres = centres.copy()
        current_score = score_centres(current_centres)
        step_size = 0.005
        no_improve_count = 0
        max_no_improve = 20
        
        for _ in range(max_iters):
            if check_time():
                break
                
            # Find the binding constraint
            constraint_type, idx1, idx2, val = get_binding_info(current_centres)
            
            if constraint_type == 'pair':
                i, j = idx1, idx2
                # Vector from i to j
                diff = current_centres[j] - current_centres[i]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random kick if coincident
                    current_centres[i] += np.random.normal(0, 0.01, 3)
                    current_centres[j] -= np.random.normal(0, 0.01, 3)
                    current_centres = np.clip(current_centres, -0.499, 0.499)
                    continue
                    
                # Unit vector
                u = diff / dist
                
                # Move i in -u direction, j in +u direction
                # We want to increase distance, so move apart
                # However, we must ensure we don't violate boundaries.
                # Simple move:
                move_i = -u * step_size
                move_j = u * step_size
                
                new_i = current_centres[i] + move_i
                new_j = current_centres[j] + move_j
                
                # Clip to box
                new_i = np.clip(new_i, -0.499, 0.499)
                new_j = np.clip(new_j, -0.499, 0.499)
                
                # Check if move improves score
                trial = current_centres.copy()
                trial[i] = new_i
                trial[j] = new_j
                
                trial_score = score_centres(trial)
                
                if trial_score > current_score:
                    current_centres = trial
                    current_score = trial_score
                    no_improve_count = 0
                    # Optionally increase step size slightly if we are making good progress
                    # step_size = min(step_size * 1.05, 0.05) 
                else:
                    no_improve_count += 1
                    # Try a smaller step or different direction? 
                    # For now, reduce step size to allow finer tuning
                    step_size *= 0.9
                    if step_size < 1e-5:
                        step_size = 0.005
                        # If we stuck, try a random perturbation of the binding points
                        if np.random.rand() < 0.1:
                            trial = current_centres.copy()
                            trial[i] += np.random.normal(0, 0.01, 3)
                            trial[j] += np.random.normal(0, 0.01, 3)
                            trial = np.clip(trial, -0.499, 0.499)
                            trial_score = score_centres(trial)
                            if trial_score > current_score:
                                current_centres = trial
                                current_score = trial_score
                                no_improve_count = 0
                                step_size = 0.005

            elif constraint_type == 'boundary':
                i, k = idx1, idx2
                # Point i is too close to face k
                # Face k: 0->-x, 1->+x, 2->-y, 3->+y, 4->-z, 5->+z
                # Actually, np.abs(centres) gives distance to nearest face.
                # If centre[i][k] > 0, it's close to +face. If < 0, close to -face.
                # We want to move it towards the center (0)
                # Direction: -sign(centre[i][k])
                val = current_centres[i][k]
                if abs(val) < 1e-10:
                    # Already at center, shouldn't be binding unless other points are close
                    # But if it is binding, it means r is limited by this point's distance to face.
                    # This implies it's at the boundary? No, score is min(dist).
                    # If it's the min, moving it inward increases this term.
                    pass
                
                move_dir = -np.sign(val)
                if move_dir == 0:
                    move_dir = 1 # arbitrary if exactly 0, but unlikely to be binding then
                    
                move = np.zeros(3)
                move[k] = move_dir * step_size
                
                new_i = current_centres[i] + move
                new_i = np.clip(new_i, -0.499, 0.499)
                
                trial = current_centres.copy()
                trial[i] = new_i
                
                trial_score = score_centres(trial)
                
                if trial_score > current_score:
                    current_centres = trial
                    current_score = trial_score
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    step_size *= 0.9
                    if step_size < 1e-5:
                        step_size = 0.005
                        if np.random.rand() < 0.1:
                            trial = current_centres.copy()
                            trial[i] += np.random.normal(0, 0.01, 3)
                            trial = np.clip(trial, -0.499, 0.499)
                            trial_score = score_centres(trial)
                            if trial_score > current_score:
                                current_centres = trial
                                current_score = trial_score
                                no_improve_count = 0
                                step_size = 0.005
            
            if no_improve_count > max_no_improve:
                break
                
        return current_centres, current_score

    def soft_min_score(centres_flat, beta=100.0):
        """
        Smooth approximation of the negative radius.
        """
        centres = centres_flat.reshape(N, 3)
        if not np.all(np.isfinite(centres)):
            return 1e10
        
        # Boundary terms
        boundary_terms = 0.5 - np.abs(centres)
        
        # Pairwise terms
        dists = cdist(centres, centres, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        iu = np.triu_indices(N, k=1)
        pair_terms = dists[iu] / 2.0
        
        all_terms = np.concatenate([boundary_terms.flatten(), pair_terms])
        
        # Soft-min
        max_term = np.max(all_terms)
        shifted = -beta * (all_terms - max_term)
        shifted = np.clip(shifted, -500, 500)
        exp_terms = np.exp(shifted)
        sum_exp = np.sum(exp_terms)
        if sum_exp == 0 or not np.isfinite(sum_exp):
            return 1e10
        soft_min_val = max_term - np.log(sum_exp) / beta
        
        return -soft_min_val

    def halton_sequence(n, base):
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
    
    def generate_halton_init(seed_offset=0):
        bases = [(2, 3, 5), (3, 5, 7), (5, 7, 11), (2, 5, 7)]
        b = bases[seed_offset % len(bases)]
        h1 = halton_sequence(N, b[0])
        h2 = halton_sequence(N, b[1])
        h3 = halton_sequence(N, b[2])
        
        scale = 0.9
        centres = np.column_stack((
            h1 * scale - scale/2,
            h2 * scale - scale/2,
            h3 * scale - scale/2
        ))
        
        rng = np.random.default_rng(42 + seed_offset)
        centres += rng.normal(0, 0.015, size=(N, 3))
        centres = np.clip(centres, -0.49, 0.49)
        return centres
    
    def generate_random_init(seed):
        rng = np.random.default_rng(seed)
        centres = rng.uniform(-0.45, 0.45, size=(N, 3))
        return centres

    def generate_grid_init():
        coords = np.linspace(-0.42, 0.42, 4)
        grid = np.array([[x, y, z] for x in coords for y in coords for z in coords])
        rng = np.random.default_rng(123)
        indices = rng.choice(len(grid), size=N, replace=False)
        centres = grid[indices].copy()
        centres += rng.normal(0, 0.01, size=(N, 3))
        centres = np.clip(centres, -0.49, 0.49)
        return centres

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    best_centres = None
    best_score = 0.0
    
    # Initializers
    initializers = [
        ("halton0", lambda: generate_halton_init(0)),
        ("halton1", lambda: generate_halton_init(1)),
        ("halton2", lambda: generate_halton_init(2)),
        ("grid", lambda: generate_grid_init()),
        ("rand1", lambda: generate_random_init(1)),
        ("rand2", lambda: generate_random_init(2)),
    ]
    
    for name, gen_func in initializers:
        if check_time():
            break
        
        try:
            centres = gen_func()
        except Exception:
            continue
        
        current_score = score_centres(centres)
        current_centres = centres.copy()
        
        # Stage 1: L-BFGS-B refinement
        # Use a few beta values to guide the optimization
        for beta in [20.0, 100.0]:
            if check_time():
                break
            
            def smooth_obj(cf, b=beta):
                return soft_min_score(cf, beta=b)
            
            try:
                result = minimize(
                    smooth_obj,
                    current_centres.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 10}
                )
            except Exception:
                continue
            
            new_centres = result.x.reshape(N, 3)
            if not np.all(np.isfinite(new_centres)):
                continue
            new_score = score_centres(new_centres)
            if new_score > current_score:
                current_centres = new_centres
                current_score = new_score
        
        # Stage 2: Targeted Local Search
        if not check_time():
            polished_centres, polished_score = local_search(current_centres, max_iters=200)
            if polished_score > current_score:
                current_centres = polished_centres
                current_score = polished_score
        
        if current_score > best_score:
            best_centres = current_centres.copy()
            best_score = current_score
    
    # Final polish on the best result found
    if best_centres is not None and not check_time():
        polished_centres, polished_score = local_search(best_centres, max_iters=500)
        if polished_score > best_score:
            best_centres = polished_centres
            best_score = polished_score

    # Fallback
    if best_centres is None:
        best_centres = generate_halton_init(0)
    
    return best_centres

# EVOLVE-BLOCK-END
