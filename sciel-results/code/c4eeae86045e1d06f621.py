"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, squareform

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin
    
    # Start with a structured grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    centres = grid[:N].copy()
    
    def compute_radius(c):
        """Compute the maximum common radius for given centres."""
        if not np.all(np.isfinite(c)):
            return 0.0
        if c.shape != (N, 3):
            return 0.0
        if np.any(np.abs(c) > 0.5 + 1e-10):
            return 0.0
        wall_min = 0.5 - np.max(np.abs(c), axis=1)
        if np.any(wall_min < -1e-10):
            return 0.0
        r_wall = np.min(wall_min)
        d = pdist(c)
        if len(d) == 0:
            r_pair = np.inf
        else:
            r_pair = np.min(d) / 2.0
        return min(r_wall, r_pair)
    
    def smooth_objective(c, beta=50.0):
        """
        Smooth approximation of the radius.
        Uses soft-min to create a differentiable objective.
        r = min(r_wall, r_pair) ≈ -1/beta * log(sum(exp(-beta*r_i)))
        """
        if not np.all(np.isfinite(c)):
            return -1e10, np.zeros((N, 3))
        
        # Wall distances
        wall_dists = 0.5 - np.abs(c)  # shape (N, 3)
        # r_wall for each point = min over 3 coords
        # r_pair for each pair = dist/2
        
        # Compute pairwise distances
        d = pdist(c)
        r_pairs = d / 2.0  # shape (N*(N-1)/2,)
        
        # Combine all constraints
        all_constraints = np.concatenate([wall_dists.flatten(), r_pairs])
        
        # Soft minimum
        min_c = np.min(all_constraints)
        max_c = np.max(all_constraints)
        range_c = max_c - min_c + 1e-10
        
        # Scaled for numerical stability
        scaled = (all_constraints - min_c) / range_c * beta
        exp_neg = np.exp(-scaled)
        sum_exp = np.sum(exp_neg)
        
        if sum_exp < 1e-30:
            return -1e10, np.zeros((N, 3))
        
        log_sum = np.log(sum_exp)
        soft_min = min_c + range_c / beta * log_sum
        
        # Gradient with respect to c
        # d(soft_min)/dc = sum_k [ -d(soft_min)/d(constraint_k) * d(constraint_k)/dc ]
        # soft_min = min_c + range_c/beta * log(sum exp(-beta*(c_k - min_c)/range_c))
        # This is complex; use automatic differentiation or finite differences
        
        # Use finite differences for gradient
        eps = 1e-7
        grad = np.zeros((N, 3))
        f0 = soft_min
        for i in range(N):
            for j in range(3):
                c_pert = c.copy()
                c_pert[i, j] += eps
                # Recompute soft_min
                wall_dists_p = 0.5 - np.abs(c_pert)
                d_p = pdist(c_pert)
                r_pairs_p = d_p / 2.0
                all_c_p = np.concatenate([wall_dists_p.flatten(), r_pairs_p])
                min_cp = np.min(all_c_p)
                maxcp = np.max(all_c_p)
                rangecp = maxcp - mincp + 1e-10
                scaled_p = (all_c_p - mincp) / rangecp * beta
                exp_neg_p = np.exp(-scaled_p)
                sum_exp_p = np.sum(exp_neg_p)
                if sum_exp_p < 1e-30:
                    f_p = -1e10
                else:
                    log_sum_p = np.log(sum_exp_p)
                    f_p = mincp + rangecp / beta * log_sum_p
                grad[i, j] = (f_p - f0) / eps
        
        return f0, grad
    
    def gradient_ascent(c, max_iters=200, lr_init=0.01):
        """Gradient ascent on the smooth objective."""
        c = c.copy()
        best_r = compute_radius(c)
        lr = lr_init
        beta = 30.0
        
        for it in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            
            f, grad = smooth_objective(c, beta)
            
            if not np.isfinite(f):
                break
            
            # Take gradient step
            c_new = c + lr * grad
            
            # Clamp to cube with small margin
            c_new = np.clip(c_new, -0.499, 0.499)
            
            # Check if improved
            r_new = compute_radius(c_new)
            if r_new > best_r + 1e-10:
                c = c_new
                best_r = r_new
                lr = min(lr * 1.02, 0.05)
            else:
                lr *= 0.9
                if lr < 1e-6:
                    break
        
        return c, best_r
    
    # Multi-start optimization
    rng = np.random.RandomState(42)
    best_centres = centres.copy()
    best_r = compute_radius(best_centres)
    
    # First: optimize the grid starting point
    c_opt, r_opt = gradient_ascent(centres, max_iters=300)
    if r_opt > best_r:
        best_r = r_opt
        best_centres = c_opt.copy()
    
    # Then: try several random restarts
    n_restarts = 15
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Generate random starting point
        if restart == 0:
            c = best_centres.copy()
        else:
            # Mix of random and perturbed
            if restart % 3 == 0:
                c = rng.uniform(-0.45, 0.45, (N, 3))
            else:
                c = best_centres + rng.normal(0, 0.02, (N, 3))
                c = np.clip(c, -0.49, 0.49)
        
        c_opt, r_opt = gradient_ascent(c, max_iters=200)
        if r_opt > best_r:
            best_r = r_opt
            best_centres = c_opt.copy()
        
        # Occasionally try a different beta for the smooth objective
        if restart % 5 == 2 and time.time() < deadline - 20.0:
            c_opt2, r_opt2 = gradient_ascent(c, max_iters=150, lr_init=0.005)
            if r_opt2 > best_r:
                best_r = r_opt2
                best_centres = c_opt2.copy()
    
    # Final validation
    if not np.all(np.isfinite(best_centres)):
        return centres
    if best_centres.shape != (N, 3):
        return centres
    if np.any(np.abs(best_centres) > 0.5 + 1e-10):
        return centres
    
    return best_centres
# EVOLVE-BLOCK-END
