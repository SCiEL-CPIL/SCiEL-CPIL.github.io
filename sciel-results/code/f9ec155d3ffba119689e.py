"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Nelder-Mead and (1+1)-ES optimizer for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin
        budget = self.budget
        dim = self.dim

        calls = 0
        best_fx = np.inf
        best_x = np.zeros(dim)
        rng = np.random.default_rng(42)

        def safe_func(x):
            nonlocal calls, best_fx, best_x
            if calls >= budget:
                return np.inf
            calls += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x_clipped))
            except Exception:
                fx = np.inf
            if np.isfinite(fx) and fx < best_fx:
                best_fx = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Nelder-Mead with restarts
        # We allocate 70% of the budget to NM
        nm_budget = int(budget * 0.7)
        
        # Run NM with potential restarts if stagnating
        # We will try up to 2 restarts if we have budget
        restarts = 0
        max_restarts = 2
        
        while calls < nm_budget and time.time() < deadline:
            # Initialize Simplex
            # Start at a random point or origin? 
            # If restarts == 0, start at origin (or best so far if restarting? No, random is better for diversity)
            if restarts == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-5, 5, dim)
            
            # Initial step size
            step = 2.0
            
            simplex = []
            f_values = []
            
            f0 = safe_func(x0)
            simplex.append(x0)
            f_values.append(f0)
            
            for i in range(dim):
                xi = x0.copy()
                xi[i] += step
                fi = safe_func(xi)
                simplex.append(xi)
                f_values.append(fi)
                
            if calls >= nm_budget:
                break

            # NM Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            n = dim + 1
            
            stagnation_counter = 0
            last_best = best_fx
            
            while calls < nm_budget and time.time() < deadline:
                # Sort
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]
                
                # Check convergence
                if f_values[-1] - f_values[0] < 1e-8:
                    break
                    
                # Check stagnation
                if best_fx == last_best:
                    stagnation_counter += 1
                else:
                    last_best = best_fx
                    stagnation_counter = 0
                    
                if stagnation_counter > 50:
                    break # Trigger restart or end phase

                x_worst = simplex[-1]
                x_best = simplex[0]
                x_centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = x_centroid + alpha * (x_centroid - x_worst)
                f_ref = safe_func(x_ref)

                if f_values[0] <= f_ref < f_values[-2]:
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
                elif f_ref < f_values[0]:
                    # Expansion
                    x_exp = x_centroid + gamma * (x_ref - x_centroid)
                    f_exp = safe_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_values[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < f_values[-1]:
                        x_con = x_centroid + rho * (x_ref - x_centroid)
                    else:
                        x_con = x_centroid - rho * (x_worst - x_centroid)
                    
                    f_con = safe_func(x_con)
                    if f_con < min(f_ref, f_values[-1]):
                        simplex[-1] = x_con
                        f_values[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, n):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            f_values[i] = safe_func(simplex[i])
                            
            # If we broke out of NM due to stagnation and have budget left, restart
            if calls < nm_budget and stagnation_counter > 50:
                if restarts < max_restarts:
                    restarts += 1
                    continue
            # If we broke out due to convergence or budget, break outer loop
            break

        # Phase 2: (1+1)-ES Polish
        # Use remaining budget
        if calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_fx
            # Start with small sigma
            sigma = 0.1
            
            while calls < budget and time.time() < deadline:
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = safe_func(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2
                else:
                    sigma *= 0.85
                    
                # If sigma becomes too small, break
                if sigma < 1e-6:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
