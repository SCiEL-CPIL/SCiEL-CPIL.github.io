"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid: coarse grid search + multi-restart L-BFGS-B for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        max_time = 290.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() - start_time > max_time:
                return float('inf')
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse grid search to find promising basins
        # Use a stratified grid with random perturbation
        grid_size = 4  # 4 points per dimension -> 4^5 = 1024, but we'll subsample
        n_grid = min(budget // 3, 100)
        
        # Create stratified grid points
        grid_points = []
        for i in range(n_grid):
            # Use Latin Hypercube sampling for better coverage
            x = np.zeros(dim)
            for d in range(dim):
                # Stratified: divide [-5, 5] into 4 strata, pick one randomly
                stratum = rng.integers(0, 4)
                x[d] = -5 + (stratum + rng.uniform(0, 1)) * 2.5
            x = np.clip(x, -5, 5)
            grid_points.append(x)
        
        # Also add the origin and some random points
        grid_points.append(np.zeros(dim))
        for _ in range(5):
            grid_points.append(rng.uniform(-5, 5, dim))
        
        # Evaluate grid points
        grid_results = []
        for x in grid_points:
            if calls >= budget or time.time() - start_time > max_time:
                break
            fx = safe_call(x)
            if fx != float('inf'):
                grid_results.append((fx, x.copy()))
        
        # Sort by fitness and keep top candidates
        grid_results.sort(key=lambda t: t[0])
        top_candidates = [x for _, x in grid_results[:min(10, len(grid_results))]]
        
        # Phase 2: Multi-restart L-BFGS-B local optimization
        # Use remaining budget for local optimization
        remaining_budget = budget - calls
        n_restarts = min(len(top_candidates), max(1, remaining_budget // 30))
        
        for i in range(n_restarts):
            if calls >= budget or time.time() - start_time > max_time:
                break
            
            start_x = top_candidates[i].copy()
            
            # Define objective function for L-BFGS-B
            def obj(x):
                return safe_call(x)
            
            # Use L-BFGS-B with bounds
            try:
                result = minimize(
                    obj,
                    start_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxls': 20}
                )
                
                # Update best if improved
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass
        
        # Phase 3: Final Nelder-Mead polishing on the best solution
        if calls < budget and time.time() - start_time > max_time - 10:
            pass  # Not enough time for NM
        
        elif calls < budget and time.time() - start_time < max_time:
            nm_budget = budget - calls
            if nm_budget > 10:
                # Build initial simplex
                simplex = [best_x.copy()]
                base_sigma = 0.3
                for i in range(dim):
                    pt = best_x.copy()
                    pt[i] += base_sigma
                    pt = np.clip(pt, -5, 5)
                    simplex.append(pt)
                
                # Evaluate simplex
                simplex_f = []
                for pt in simplex:
                    if calls >= budget or time.time() - start_time > max_time:
                        break
                    f = safe_call(pt)
                    simplex_f.append(f)
                
                # Nelder-Mead parameters
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma_shrink = 0.5
                
                nm_iter = 0
                max_nm_iter = min(nm_budget - 1, 50)
                
                while nm_iter < max_nm_iter and calls < budget and time.time() - start_time < max_time:
                    nm_iter += 1
                    
                    # Sort simplex by fitness
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]
                    
                    # Check convergence
                    if simplex_f[-1] - simplex_f[0] < 1e-10:
                        break
                    
                    # Centroid of all but worst point
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    worst_pt = simplex[-1]
                    worst_f = simplex_f[-1]
                    x_ref = centroid + alpha * (centroid - worst_pt)
                    x_ref = np.clip(x_ref, -5, 5)
                    
                    if calls >= budget or time.time() - start_time > max_time:
                        break
                    f_ref = safe_call(x_ref)
                    
                    if f_ref < simplex_f[0]:
                        # Expansion
                        x_exp = centroid + gamma * (x_ref - centroid)
                        x_exp = np.clip(x_exp, -5, 5)
                        if calls >= budget or time.time() - start_time > max_time:
                            break
                        f_exp = safe_call(x_exp)
                        if f_exp < f_ref:
                            x_new, f_new = x_exp, f_exp
                        else:
                            x_new, f_new = x_ref, f_ref
                        simplex[-1] = x_new
                        simplex_f[-1] = f_new
                        if f_new < best_f:
                            best_f = f_new
                            best_x = x_new.copy()
                    elif f_ref < simplex_f[-2]:
                        # Accept reflection
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                        if f_ref < best_f:
                            best_f = f_ref
                            best_x = x_ref.copy()
                    else:
                        # Contraction
                        x_con = centroid + rho * (worst_pt - centroid)
                        x_con = np.clip(x_con, -5, 5)
                        if calls >= budget or time.time() - start_time > max_time:
                            break
                        f_con = safe_call(x_con)
                        if f_con < worst_f:
                            simplex[-1] = x_con
                            simplex_f[-1] = f_con
                            if f_con < best_f:
                                best_f = f_con
                                best_x = x_con.copy()
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma_shrink * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                if calls >= budget or time.time() - start_time > max_time:
                                    break
                                f = safe_call(simplex[i])
                                simplex_f[i] = f
                                if f < best_f:
                                    best_f = f
                                    best_x = simplex[i].copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
