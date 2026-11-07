"""Optimized BBOB f14 solver with Multi-Start Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start Nelder-Mead optimizer for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 310.0  # Safety margin before 330s kill

        rng = np.random.default_rng(42)
        
        # State tracking
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Helper to safely evaluate function and update best
        def evaluate(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse Exploration
        # We use a significant portion of the budget (e.g., 20%) for random sampling
        # to find a good starting point for the local optimizer.
        # BBOB functions are often shifted and rotated, so a good initial point
        # is crucial for Nelder-Mead to converge to the global minimum (or a deep local one).
        n_coarse = max(10, int(budget * 0.2))
        
        # Always evaluate origin first as a sanity check, though random is better
        # for shifted instances.
        x0 = np.zeros(dim)
        f0 = evaluate(x0)
        
        for _ in range(n_coarse - 1):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        # Phase 2: Nelder-Mead Refinement
        # We run Nelder-Mead starting from the best point found in Phase 1.
        # We allocate the remaining budget to this phase.
        
        # Nelder-Mead parameters
        # Standard coefficients
        alpha = 1.0    # Reflection
        gamma = 2.0    # Expansion
        rho = 0.5      # Contraction
        sigma = 0.5    # Shrink
        
        # Initialize simplex around best_x
        # The size of the initial simplex is critical.
        # If too large, it may jump out of the basin. If too small, it converges slowly.
        # A step size of 0.1 to 0.5 is typical for [-5, 5] box.
        initial_step = 0.5
        
        simplex = []
        simplex_f = []
        
        # First vertex is the best point found
        simplex.append(best_x.copy())
        simplex_f.append(best_f)
        
        # Generate other vertices by perturbing each dimension
        for i in range(dim):
            v = best_x.copy()
            # Perturb dimension i by initial_step
            # Use a deterministic direction to ensure coverage
            v[i] += initial_step
            # Clip to box
            v = np.clip(v, -5, 5)
            fv = evaluate(v)
            simplex.append(v)
            simplex_f.append(fv)
        
        # Convert to arrays for vectorized operations where possible
        # But Nelder-Mead is inherently sequential in its updates.
        
        # Main Nelder-Mead Loop
        # We continue until budget or time is exhausted.
        # Each iteration of the standard NM algorithm costs 1 function evaluation.
        
        while calls_made < budget and time.time() < deadline:
            # Sort vertices by function value
            # We need to find the worst (highest f) and second worst
            # and the best (lowest f)
            
            # For efficiency with small N (dim=5), simple sorting is fine
            idx = np.argsort(simplex_f)
            simplex = [simplex[i] for i in idx]
            simplex_f = [simplex_f[i] for i in idx]
            
            # Check for convergence
            # If the spread of function values is very small, we are likely at a minimum
            f_spread = simplex_f[-1] - simplex_f[0]
            if f_spread < 1e-8:
                # Converged, try shrinking or break if we can't improve
                # For robustness, we might just break if we've spent enough time
                # but let's try to shrink the simplex to ensure precision
                # Actually, if spread is tiny, we are stuck. 
                # We can break to save time for other restarts if we had them, 
                # but here we are in a single NM run. 
                # Let's break if we can't make progress.
                break
            
            # Identify worst and second worst
            x_worst = simplex[-1]
            f_worst = simplex_f[-1]
            x_second_worst = simplex[-2]
            f_second_worst = simplex_f[-2]
            x_best = simplex[0]
            f_best = simplex_f[0]
            
            # Centroid of all points except the worst
            x_c = np.sum(simplex[:-1], axis=0) / dim
            
            # 1. Reflection
            x_r = x_c + alpha * (x_c - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = evaluate(x_r)
            
            if f_r < f_best:
                # 2. Expansion
                x_e = x_c + gamma * (x_r - x_c)
                x_e = np.clip(x_e, -5, 5)
                f_e = evaluate(x_e)
                
                if f_e < f_r:
                    # Replace worst with expanded point
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    # Replace worst with reflected point
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < f_second_worst:
                # Accept reflection
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # 3. Contraction
                # Check if f_r is better than worst but not better than second worst
                # We need to decide between outside and inside contraction
                if f_r < f_worst:
                    # Outside contraction
                    x_c_out = x_c + 0.5 * (x_r - x_c)
                    x_co = x_c_out
                else:
                    # Inside contraction
                    x_co = x_c + 0.5 * (x_worst - x_c)
                
                x_co = np.clip(x_co, -5, 5)
                f_co = evaluate(x_co)
                
                if f_co < f_worst:
                    # Accept contraction
                    simplex[-1] = x_co
                    simplex_f[-1] = f_co
                else:
                    # 4. Shrink
                    # Shrink all points except the best towards the best
                    for i in range(1, dim + 1):
                        simplex[i] = x_best + sigma * (simplex[i] - x_best)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = evaluate(simplex[i])
                        
                        # If budget runs out during shrink, we just stop
                        if calls_made >= budget:
                            break
                    
                    if calls_made >= budget:
                        break

        # Ensure best_x is the absolute best found
        if best_x is None:
             best_x = np.zeros(dim)
             best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
