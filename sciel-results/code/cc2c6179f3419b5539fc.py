import numpy as np
import time

class Optimizer:
    """
    Hybrid Optimizer: Global Latin Hypercube Search followed by Nelder-Mead Local Refinement.
    Designed for BBOB f13 (Sharp Ridge) in 5D with budget 500.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # Safety margin before 300s/330s limit
        
        calls = 0
        best_x = None
        best_f = float('inf')
        
        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                x = np.clip(x, -5.0, 5.0)
                val = func(x)
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val
            except Exception:
                return None

        # Phase 1: Global Exploration using Latin Hypercube Sampling
        # We allocate ~70% of the budget to global search to ensure we find the ridge.
        n_global = int(budget * 0.7)
        n_local = budget - n_global
        
        if n_global < dim + 1:
            n_global = dim + 1
            n_local = budget - n_global
            
        # Generate LHS points
        # We want points in [-5, 5]^dim
        # LHS: Divide each dimension into n_global bins, pick one random point per bin in each dimension, then shuffle.
        rng = np.random.default_rng(42)
        
        # Create bins for each dimension
        # points[i, j] is the j-th point in the i-th dimension
        # We generate n_global points
        lhs_points = np.zeros((n_global, dim))
        for d in range(dim):
            # Divide [-5, 5] into n_global bins
            bin_width = 10.0 / n_global
            # Random offset in each bin
            offsets = rng.random(n_global) * bin_width
            # Center of each bin + offset
            # Bin centers: -5 + (i + 0.5) * bin_width
            centers = -5.0 + (np.arange(n_global) + 0.5) * bin_width
            lhs_points[:, d] = centers + offsets
        
        # Shuffle the dimensions independently to break correlation
        # Actually, standard LHS construction:
        # For each dimension d, we have a set of n_global values.
        # We need to permute the rows for each column? No.
        # Standard LHS:
        # 1. For each dimension d, generate n_global values in [0, 1] by taking (i + u_i) / n_global.
        # 2. Map to [-5, 5].
        # 3. For each dimension d, permute the column.
        
        lhs_points = np.zeros((n_global, dim))
        for d in range(dim):
            # Generate values in [0, 1]
            u = rng.random(n_global)
            vals = (np.arange(n_global) + u) / n_global
            # Map to [-5, 5]
            lhs_points[:, d] = -5.0 + 10.0 * vals
            # Shuffle the column to create the hypercube property
            perm = rng.permutation(n_global)
            lhs_points[:, d] = lhs_points[perm, d]
            
        # Evaluate global points
        global_results = []
        for i in range(n_global):
            if calls >= budget:
                break
            x = lhs_points[i]
            f = safe_call(x)
            if f is not None:
                global_results.append((f, x))
                
        if not global_results:
            return best_x, best_f
            
        # Sort global results by function value
        global_results.sort(key=lambda x: x[0])
        
        # Phase 2: Local Refinement using Nelder-Mead
        # We will try to refine the top few candidates from the global search.
        # Start with the best one.
        
        # Nelder-Mead Parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # We have n_local calls left.
        # We will run NM for the best candidate.
        # If it stagnates, we might try the second best, but budget is tight.
        # Let's just run NM on the best point found so far (which is global_results[0][1]).
        
        start_x = global_results[0][1].copy()
        start_f = global_results[0][0]
        
        # Initialize Simplex
        # Step size: Start with a moderate step, e.g., 0.5
        # If the ridge is sharp, we might need to be careful.
        # Let's try 0.5 initially.
        step_size = 0.5
        
        simplex = [start_x.copy()]
        f_values = [start_f]
        
        for i in range(dim):
            x_new = start_x.copy()
            x_new[i] += step_size
            f_new = safe_call(x_new)
            if f_new is None:
                # Out of budget or time
                return best_x, best_f
            simplex.append(x_new)
            f_values.append(f_new)
            
        if len(simplex) < dim + 1:
            return best_x, best_f
            
        simplex = np.array(simplex)
        f_values = np.array(f_values)
        
        # Main NM Loop
        stagnant_count = 0
        max_stagnant = 10 # Restart or shrink if no improvement for 10 iterations
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Sort simplex
            idx = np.argsort(f_values)
            simplex = simplex[idx]
            f_values = f_values[idx]
            
            # Update global best
            if f_values[0] < best_f:
                best_f = f_values[0]
                best_x = simplex[0].copy()
                
            # Check for convergence
            spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
            if np.max(spread) < 1e-8 or np.max(f_values) - np.min(f_values) < 1e-12:
                # Converged or stuck in a tiny region
                # Try expanding or restarting with a larger step?
                # For now, break and try another candidate if we have time/budget
                break
                
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5.0, 5.0)
            fr = safe_call(xr)
            if fr is None:
                break
                
            if fr < f_values[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5.0, 5.0)
                fe = safe_call(xe)
                if fe is None:
                    break
                if fe < fr:
                    simplex[-1] = xe
                    f_values[-1] = fe
                else:
                    simplex[-1] = xr
                    f_values[-1] = fr
            elif fr < f_values[-2]:
                # Reflection
                simplex[-1] = xr
                f_values[-1] = fr
            else:
                # Contraction
                if fr < f_values[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5.0, 5.0)
                fc = safe_call(xc)
                if fc is None:
                    break
                if fc <= min(fr, f_values[-1]):
                    simplex[-1] = xc
                    f_values[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fi = safe_call(simplex[i])
                        if fi is None:
                            break
                        f_values[i] = fi
                    if fi is None:
                        break
                        
            # Stagnation check
            if f_values[0] >= best_f - 1e-10: # No improvement in global best
                stagnant_count += 1
            else:
                stagnant_count = 0
                
            if stagnant_count > max_stagnant:
                # Stagnant. Break and try next best global candidate if available
                # Or just break. Given tight budget, breaking is safer.
                # We could try to restart NM with a larger step from best_x
                # But let's just break and rely on the global search having found the best spot.
                # If we have budget left, we could do a random restart near best_x.
                # Let's try a few random perturbations if we have time.
                if calls < budget and time.time() < deadline:
                    # Do 5 random perturbations around best_x
                    for _ in range(5):
                        if calls >= budget:
                            break
                        if time.time() > deadline:
                            break
                        x_rand = best_x + rng.normal(0, 0.1, size=dim)
                        safe_call(x_rand)
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
