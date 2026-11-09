"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-start Nelder-Mead optimizer for BBOB f19 (Griewank-Rosenbrock).
    Strategy:
    1. Use a fixed set of diverse starting points (center + random).
    2. Run Nelder-Mead simplex from each starting point until budget or convergence.
    3. Nelder-Mead is fast (1 eval per vertex update) and effective for smooth functions.
    4. Early progress is maximized by quickly converging on good basins.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # 300s limit, keep 30s safety margin
        deadline = start_time + 270.0
        
        dim = self.dim
        budget = self.budget
        
        global_best_x = None
        global_best_f = float('inf')
        calls_made = 0
        
        def call_func(x):
            nonlocal calls_made, global_best_x, global_best_f
            if calls_made >= budget:
                return global_best_f
            val = func(x)
            calls_made += 1
            if val < global_best_f:
                global_best_f = val
                global_best_x = x.copy()
            return val

        # --- Phase 1: Generate diverse starting points ---
        # Start with center
        starts = [np.zeros(dim)]
        
        # Add random points
        # We want enough starts to cover the space but not too many to waste budget on poor basins.
        # 500 budget / ~100 evals per NM run (approx) = 5 starts?
        # But NM can be faster. Let's aim for ~5-10 starts.
        # Let's use 8 random starts.
        num_random_starts = 8
        for _ in range(num_random_starts):
            x_rand = np.random.uniform(-5, 5, dim)
            starts.append(x_rand)
        
        # --- Phase 2: Nelder-Mead from each start ---
        for start_x in starts:
            if time.time() > deadline or calls_made >= budget:
                break
            
            # Initialize simplex
            # Vertices: start_x + delta * e_i for i in range(dim), and start_x itself
            # Standard NM simplex size is usually 1.0 or 0.05 * range.
            # Let's use 0.5 as initial simplex size (range is 10, so 0.5 is 5% of range).
            delta = 0.5
            simplex = [start_x.copy()]
            for i in range(dim):
                v = start_x.copy()
                v[i] += delta
                simplex.append(v)
            
            # Evaluate initial simplex
            fvals = []
            for v in simplex:
                if time.time() > deadline or calls_made >= budget:
                    break
                fvals.append(call_func(v))
            
            if calls_made >= budget or time.time() > deadline:
                break
                
            # Ensure we have dim+1 values
            if len(fvals) < dim + 1:
                continue
            
            # Nelder-Mead Loop
            # Parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            
            max_iter = 100  # Safety limit per NM run
            
            for _ in range(max_iter):
                if time.time() > deadline or calls_made >= budget:
                    break
                
                # Sort simplex by f-values
                indices = np.argsort(fvals)
                simplex = [simplex[i] for i in indices]
                fvals = [fvals[i] for i in indices]
                
                # Check convergence
                # If range of fvals is small, stop
                if fvals[-1] - fvals[0] < 1e-10:
                    break
                # Also check if simplex size is small
                # dist = np.max(np.linalg.norm(np.array(simplex) - np.mean(simplex, axis=0), axis=1))
                # if dist < 1e-6: break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = call_func(x_ref)
                
                if fvals[0] <= f_ref < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                elif f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = call_func(x_exp)
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < fvals[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (centroid - simplex[-1])
                    
                    x_con = np.clip(x_con, -5, 5)
                    f_con = call_func(x_con)
                    
                    if f_con < min(f_ref, fvals[-1]):
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = call_func(simplex[i])
                        
                        if calls_made >= budget or time.time() > deadline:
                            break

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
