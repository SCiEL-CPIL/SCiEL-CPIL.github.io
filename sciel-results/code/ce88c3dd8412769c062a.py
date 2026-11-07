"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with adaptive restarts and deadline guard for sharp ridge f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Track best found
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            return fx

        # Phase 1: Broad random sampling to find a good starting point
        # Use 10% of budget for initial exploration
        explore_budget = max(10, budget // 10)
        for _ in range(explore_budget):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            # Fallback if no calls were made (shouldn't happen)
            x = np.zeros(dim)
            fx = safe_call(x)
            if fx is not None:
                best_f = fx
                best_x = x
        
        # Phase 2: Nelder-Mead optimization with restarts
        remaining_budget = budget - calls_made
        max_restarts = 3
        restart_count = 0
        
        while calls_made < budget and time.time() < deadline and restart_count < max_restarts:
            # Initialize simplex
            simplex = []
            simplex_f = []
            
            # First point is the current best
            simplex.append(best_x.copy())
            simplex_f.append(best_f)
            
            # Other points: perturb best_x
            base_step = 0.5
            for i in range(dim):
                x = best_x.copy()
                x[i] += base_step * rng.standard_normal()
                x = np.clip(x, -5.0, 5.0)
                fx = safe_call(x)
                if fx is None:
                    break
                simplex.append(x)
                simplex_f.append(fx)
            
            if len(simplex) < 2:
                break
                
            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            # Nelder-Mead iterations
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink
            stagnation = 0
            max_stagnation = 15
            no_improve_count = 0
            best_in_nm = best_f
            
            while calls_made < budget and time.time() < deadline:
                n = len(simplex)
                if n < 2:
                    break
                    
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = simplex_f[-1]
                
                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = safe_call(x_r)
                if f_r is None:
                    break
                
                if f_r < simplex_f[0]:
                    # Try expansion
                    x_e = centroid + gamma * (centroid - worst)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = safe_call(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    stagnation = 0
                    no_improve_count = 0
                    if simplex_f[-1] < best_in_nm:
                        best_in_nm = simplex_f[-1]
                        if simplex_f[-1] < best_f:
                            best_f = simplex_f[-1]
                            best_x = simplex[-1].copy()
                elif f_r < worst_f:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                    stagnation = 0
                    no_improve_count = 0
                    if simplex_f[-1] < best_in_nm:
                        best_in_nm = simplex_f[-1]
                        if simplex_f[-1] < best_f:
                            best_f = simplex_f[-1]
                            best_x = simplex[-1].copy()
                else:
                    # Contraction
                    x_c = centroid + rho * (worst - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = safe_call(x_c)
                    if f_c is None:
                        break
                    if f_c < worst_f:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                        stagnation = 0
                        no_improve_count = 0
                        if simplex_f[-1] < best_in_nm:
                            best_in_nm = simplex_f[-1]
                            if simplex_f[-1] < best_f:
                                best_f = simplex_f[-1]
                                best_x = simplex[-1].copy()
                    else:
                        # Shrink
                        for i in range(1, n):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_s = safe_call(simplex[i])
                            if f_s is None:
                                break
                            simplex_f[i] = f_s
                        if f_s is None:
                            break
                        stagnation += 1
                        no_improve_count += 1
                        
                        if simplex_f[0] < best_in_nm:
                            best_in_nm = simplex_f[0]
                            if simplex_f[0] < best_f:
                                best_f = simplex_f[0]
                                best_x = simplex[0].copy()
                        
                        if stagnation > max_stagnation:
                            break
                    
                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Check if simplex has collapsed
                spread = np.max(np.abs(simplex - simplex[0]), axis=0)
                if np.max(spread) < 1e-6:
                    break
                
                # Update best
                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()
            
            # Check for restart
            if calls_made < budget and time.time() < deadline:
                # If no improvement in a while, restart with a larger perturbation
                if no_improve_count > 20 or stagnation > max_stagnation:
                    restart_count += 1
                    # Perturb best_x significantly
                    perturbation = rng.standard_normal(dim) * 1.0
                    new_x = best_x + perturbation
                    new_x = np.clip(new_x, -5.0, 5.0)
                    new_f = safe_call(new_x)
                    if new_f is not None:
                        if new_f < best_f:
                            best_f = new_f
                            best_x = new_x
                    # Continue to next restart iteration
                else:
                    # No restart needed, but if we're out of budget, stop
                    if calls_made >= budget:
                        break
                    # If we're making progress, continue NM (but we already broke out of the inner loop)
                    # So we just break out of the outer loop too
                    break
            else:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
