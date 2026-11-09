"""Rewritten optimizer using Multi-Start Nelder-Mead with Global Sampling."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-Start Nelder-Mead based black-box continuous optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def eval_func(x):
            nonlocal calls_used
            calls_used += 1
            val = func(x)
            return val

        # Phase 1: Global Exploration using Latin Hypercube Sampling
        # Use 50% of the budget for global search to find good basins
        global_budget = int(budget * 0.5)
        n_global = min(global_budget, 100) # Cap global samples to avoid wasting too much
        
        # Generate Latin Hypercube Samples in [-5, 5]^d
        # For simplicity and robustness, we use random sampling but structured
        # LHS: divide each dimension into n bins, pick one random point per bin per dimension, then shuffle
        lhs_points = np.empty((n_global, d))
        for i in range(n_global):
            for j in range(d):
                # Random point in bin [i/n, (i+1)/n] scaled to [-5, 5]
                bin_start = -5.0 + (10.0 / n_global) * i
                bin_end = -5.0 + (10.0 / n_global) * (i + 1)
                lhs_points[i, j] = rng.uniform(bin_start, bin_end)
        
        # Shuffle the assignments for each dimension to decorrelate
        for j in range(d):
            perm = rng.permutation(n_global)
            lhs_points[:, j] = lhs_points[perm, j]

        # Evaluate global samples
        global_results = []
        for i in range(n_global):
            if calls_used >= budget:
                break
            x = lhs_points[i]
            f = eval_func(x)
            global_results.append((f, x))
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Sort global results to find top candidates for local search
        global_results.sort(key=lambda item: item[0])
        
        # Phase 2: Local Refinement using Nelder-Mead
        # Select top K candidates from global search
        K = 3 # Number of local restarts
        
        for k in range(K):
            if k >= len(global_results):
                break
            
            # Start Nelder-Mead from this point
            x0 = global_results[k][1].copy()
            
            # Check if we have enough budget left for a meaningful local search
            # Nelder-Mead typically needs ~ 10 * d iterations to converge
            # Each iteration costs d+1 evaluations
            # Let's reserve at least 50 evaluations for local search
            if budget - calls_used < 50:
                break
            
            # Initialize Simplex
            # Standard Nelder-Mead initialization
            simplex = np.empty((d + 1, d))
            simplex[0] = x0
            for i in range(1, d + 1):
                simplex[i] = x0.copy()
                simplex[i, i - 1] += 0.1 # Small step in direction i
            
            # Clip to bounds
            simplex = np.clip(simplex, -5, 5)
            
            # Evaluate initial simplex
            f_simplex = np.empty(d + 1)
            for i in range(d + 1):
                if calls_used >= budget:
                    break
                f_simplex[i] = eval_func(simplex[i])
                if f_simplex[i] < best_f:
                    best_f = f_simplex[i]
                    best_x = simplex[i].copy()
            
            if calls_used >= budget:
                break

            # Nelder-Mead Iterations
            alpha = 1.0 # Reflection
            gamma = 2.0 # Expansion
            rho = 0.5   # Contraction
            sigma = 0.5 # Shrinkage
            
            max_iter = (budget - calls_used) // (d + 1)
            
            for _ in range(max_iter):
                if calls_used + d + 1 > budget:
                    break
                
                # Sort simplex by fitness
                idx = np.argsort(f_simplex)
                simplex = simplex[idx]
                f_simplex = f_simplex[idx]
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = eval_func(x_ref)
                
                if f_simplex[0] <= f_ref < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    f_simplex[-1] = f_ref
                elif f_ref < f_simplex[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = eval_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_simplex[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_simplex[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < f_simplex[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (x_ref - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = eval_func(x_con)
                    
                    if f_con < min(f_ref, f_simplex[-1]):
                        simplex[-1] = x_con
                        f_simplex[-1] = f_con
                    else:
                        # Shrinkage
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = eval_func(simplex[i])
                
                # Update global best
                current_min_f = np.min(f_simplex)
                if current_min_f < best_f:
                    best_f = current_min_f
                    best_x = simplex[np.argmin(f_simplex)].copy()
                    
                # Stopping criterion: Simplex size is small
                simplex_size = np.max(np.abs(simplex - simplex[0]), axis=0)
                if np.max(simplex_size) < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
