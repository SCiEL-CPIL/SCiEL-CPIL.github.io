"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead simplex with robust global search and adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Phase 1: Global Search
        # Use 50 random points plus some structured points to find a good basin
        n_global = 50
        global_points = rng.uniform(-5, 5, size=(n_global, dim))
        
        structured = np.array([
            [0.0]*dim,
            [1.0]*dim,
            [-1.0]*dim,
            [5.0]*dim,
            [-5.0]*dim,
            [2.0]*dim,
            [-2.0]*dim
        ])
        
        all_points = np.vstack([global_points, structured])
        perm = rng.permutation(len(all_points))
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        for i in perm:
            if calls_made >= budget:
                break
            x = np.clip(all_points[i], -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if best_x is None:
            return np.zeros(dim), np.inf

        # Phase 2: Nelder-Mead with Restarts
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        current_step = 0.5
        max_restarts = 5
        restart_count = 0
        
        while calls_made < budget and restart_count <= max_restarts:
            # Initialize simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = best_x
            for i in range(1, dim + 1):
                simplex[i] = best_x.copy()
                simplex[i, i - 1] += current_step
                
            simplex_f = np.zeros(dim + 1)
            for i in range(dim + 1):
                if calls_made >= budget:
                    simplex_f[i] = np.inf
                else:
                    simplex_f[i] = func(simplex[i])
                    calls_made += 1
                    if simplex_f[i] < best_f:
                        best_f = simplex_f[i]
                        best_x = simplex[i].copy()
                        
            no_improve = 0
            max_no_improve = 15
            
            while calls_made < budget:
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]
                
                if np.max(simplex_f) - np.min(simplex_f) < 1e-8:
                    break
                    
                centroid = simplex[:-1].mean(axis=0)
                
                if calls_made >= budget: break
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_made += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
                    
                if f_r < simplex_f[0]:
                    if calls_made >= budget: break
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_made += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    no_improve = 0
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                    no_improve = 0
                else:
                    if calls_made >= budget: break
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_made += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                    
                    if f_c < f_r:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                        no_improve = 0
                    else:
                        no_improve += 1
                        for i in range(1, dim + 1):
                            if calls_made >= budget: break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = func(simplex[i])
                            calls_made += 1
                            if simplex_f[i] < best_f:
                                best_f = simplex_f[i]
                                best_x = simplex[i].copy()
                                
                if no_improve > max_no_improve:
                    # Restart with a smaller step size and perturbed center
                    restart_count += 1
                    current_step = max(0.1, current_step * 0.5)
                    best_x = best_x + rng.normal(0, 0.1, size=dim)
                    best_x = np.clip(best_x, -5, 5)
                    break # Break inner while, go to outer while
            
        return best_x, best_f
# EVOLVE-BLOCK-END
