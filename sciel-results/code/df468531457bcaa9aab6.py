"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead with adaptive restarts for BBOB f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        dim = self.dim
        budget = self.budget
        deadline = self.start_time + 290  # 10s safety margin
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # We'll do multiple Nelder-Mead runs with different starting points
        # Each run uses a portion of the budget
        # Strategy: First do a coarse random search to find good basins,
        # then refine with Nelder-Mead
        
        # Phase 1: Coarse random search with local refinement
        # Use 100 evaluations for initial exploration
        n_coarse = min(100, budget)
        candidates = []
        
        for i in range(n_coarse):
            if time.time() > deadline:
                break
            x = rng.uniform(-4, 4, dim)  # Minimizer is in [-4,4]^5
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            candidates.append((fx, x))
        
        if calls_made >= budget:
            return best_x, best_f
        
        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Nelder-Mead refinement from top candidates
        # Use remaining budget for NM runs
        remaining = budget - calls_made
        
        # Select top 5 candidates for NM refinement
        n_nm_starts = min(5, len(candidates))
        nm_budget_each = max(20, remaining // max(1, n_nm_starts))
        
        for idx in range(n_nm_starts):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break
            
            start_x = candidates[idx][1].copy()
            start_f = candidates[idx][0]
            
            # Nelder-Mead implementation
            # Initial simplex: start_x plus small perturbations
            simplex = [start_x.copy()]
            simplex_f = [start_f]
            
            # Add dim more points with small step
            step = 0.5
            for d in range(dim):
                x_new = start_x.copy()
                x_new[d] += step
                x_new = np.clip(x_new, -5, 5)
                fx_new = func(x_new)
                calls_made += 1
                simplex.append(x_new)
                simplex_f.append(fx_new)
                
                if time.time() > deadline or calls_made >= budget:
                    break
            
            # Nelder-Mead main loop
            nm_calls = dim + 1  # Initial simplex evaluations
            max_nm_iter = nm_budget_each - nm_calls
            if max_nm_iter < 1:
                continue
            
            for it in range(max_nm_iter):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break
                
                # Sort simplex by fitness
                order = np.argsort(simplex_f)
                simplex = [simplex[i].copy() for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Update best
                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()
                
                # Centroid of all points except worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_reflect = centroid + alpha * (centroid - simplex[-1])
                x_reflect = np.clip(x_reflect, -5, 5)
                f_reflect = func(x_reflect)
                calls_made += 1
                nm_calls += 1
                
                if simplex_f[0] <= f_reflect < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_reflect
                    simplex_f[-1] = f_reflect
                    continue
                elif f_reflect < simplex_f[0]:
                    # Expansion
                    gamma = 2.0
                    x_expand = centroid + gamma * (centroid - simplex[-1])
                    x_expand = np.clip(x_expand, -5, 5)
                    f_expand = func(x_expand)
                    calls_made += 1
                    nm_calls += 1
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        simplex_f[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        simplex_f[-1] = f_reflect
                    continue
                else:
                    # Contraction
                    rho = 0.5
                    if f_reflect < simplex_f[-1]:
                        # Outside contraction
                        x_contract = centroid + rho * (x_reflect - centroid)
                    else:
                        # Inside contraction
                        x_contract = centroid - rho * (x_reflect - centroid)
                    x_contract = np.clip(x_contract, -5, 5)
                    f_contract = func(x_contract)
                    calls_made += 1
                    nm_calls += 1
                    
                    if f_contract < min(f_reflect, simplex_f[-1]):
                        simplex[-1] = x_contract
                        simplex_f[-1] = f_contract
                        continue
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            f_shrink = func(simplex[i])
                            calls_made += 1
                            nm_calls += 1
                            simplex_f[i] = f_shrink
                            if time.time() > deadline or calls_made >= budget:
                                break
                        if time.time() > deadline or calls_made >= budget:
                            break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
