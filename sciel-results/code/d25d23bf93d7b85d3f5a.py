"""Hybrid Nelder-Mead + Adaptive Coordinate Descent optimizer for BBOB f6."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: Global Nelder-Mead followed by local Adaptive Coordinate Descent."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            f_val = func(x)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Global Exploration
        # Sample 20 random points to find a good starting basin
        init_points = []
        for _ in range(20):
            x = rng.uniform(-5, 5, dim)
            f = safe_call(x)
            init_points.append((x, f))
        
        if not init_points:
            return best_x, best_f

        # Pick the best point as the start for Nelder-Mead
        best_init = min(init_points, key=lambda t: t[1])
        x0 = best_init[0]
        f0 = best_init[1]
        
        # Phase 2: Nelder-Mead for local convergence (handles rotation)
        x_nm, f_nm = self._nelder_mead(x0, dim, safe_call, deadline)
        
        if f_nm < best_f:
            best_x = x_nm
            best_f = f_nm
        else:
            # If NM failed to improve, use the best random point
            x_nm = best_x
            f_nm = best_f

        # Phase 3: Adaptive Coordinate Descent for final polishing
        # This is the "Derandomized Coordinate Descent" requested
        if calls < budget and time.time() < deadline:
            x_cd, f_cd = self._adaptive_cd(x_nm, dim, safe_call, deadline)
            
            if f_cd < best_f:
                best_x = x_cd
                best_f = f_cd

        return best_x, best_f

    def _nelder_mead(self, x0, dim, safe_call, deadline):
        # Initialize simplex with dim+1 points
        simplex = [x0.copy()]
        f_simplex = [safe_call(x0)]
        
        # Generate other vertices with small perturbations
        for i in range(dim):
            x_new = x0.copy()
            # Perturb coordinate i
            step = 0.5 if x0[i] == 0 else 0.1 * x0[i]
            x_new[i] += step
            simplex.append(x_new)
            f_simplex.append(safe_call(x_new))
            
        # Sort simplex by fitness
        indices = np.argsort(f_simplex)
        simplex = [simplex[i] for i in indices]
        f_simplex = [f_simplex[i] for i in indices]
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        max_iter = 200 # Limit iterations to save budget
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Check convergence
            f_range = max(f_simplex) - min(f_simplex)
            if f_range < 1e-7:
                break
                
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            f_r = safe_call(x_r)
            
            if f_r < f_simplex[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                f_e = safe_call(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_simplex[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
            elif f_r < f_simplex[-2]:
                # Reflection is good
                simplex[-1] = x_r
                f_simplex[-1] = f_r
            else:
                # Contraction
                if f_r < f_simplex[-1]:
                    # Outside contraction
                    x_c = centroid + 0.5 * (x_r - centroid)
                    f_c = safe_call(x_c)
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_simplex[i] = safe_call(simplex[i])
                else:
                    # Inside contraction
                    x_c = centroid - 0.5 * (centroid - simplex[-1])
                    f_c = safe_call(x_c)
                    if f_c < f_simplex[-1]:
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_simplex[i] = safe_call(simplex[i])
            
            # Re-sort
            indices = np.argsort(f_simplex)
            simplex = [simplex[i] for i in indices]
            f_simplex = [f_simplex[i] for i in indices]
            
            # Early stop if best fitness hasn't improved in last 50 iterations? 
            # Keep it simple: run until max_iter or convergence.

        return simplex[0], f_simplex[0]

    def _adaptive_cd(self, x0, dim, safe_call, deadline):
        x = x0.copy()
        f = safe_call(x)
        steps = np.ones(dim) * 0.1
        min_step = 1e-5
        max_step = 10.0
        
        # Adaptive Coordinate Descent
        # We iterate through coordinates, adjusting step size based on success
        while time.time() < deadline:
            improved = False
            all_small = True
            for i in range(dim):
                if time.time() > deadline:
                    break
                
                if steps[i] > min_step * 1.1:
                    all_small = False

                # Try + step
                x_new = x.copy()
                x_new[i] += steps[i]
                f_new = safe_call(x_new)
                
                if f_new < f:
                    x = x_new
                    f = f_new
                    steps[i] = min(steps[i] * 1.5, max_step)
                    improved = True
                    continue
                
                # Try - step
                x_new = x.copy()
                x_new[i] -= steps[i]
                f_new = safe_call(x_new)
                
                if f_new < f:
                    x = x_new
                    f = f_new
                    steps[i] = min(steps[i] * 1.5, max_step)
                    improved = True
                    continue
                
                # No improvement, shrink step
                steps[i] = max(steps[i] * 0.5, min_step)

            if not improved:
                # If no coordinate improved, we might be converged or stuck
                # If all steps are minimal, stop
                if all_small:
                    break
                # Otherwise, continue (might oscillate, but steps shrink)
                # To prevent infinite loop on flat regions, we can break if steps are small
                if np.all(steps <= 1e-4):
                    break
        return x, f
# EVOLVE-BLOCK-END
