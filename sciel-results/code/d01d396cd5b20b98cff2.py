"""Rewritten optimizer using Nelder-Mead with random initialization and restarts."""
import numpy as np


class Optimizer:
    """Nelder-Mead style continuous optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls_made = 0
        best_x = None
        best_fx = np.inf

        # Helper to safely call func and track budget
        def safe_call(x):
            nonlocal calls_made, best_x, best_fx
            if calls_made >= budget:
                return np.inf
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x_clipped.copy()
            return fx

        # Phase 1: Random Initialization to find a good starting point
        # Use 10% of budget for random sampling
        init_budget = max(10, budget // 10)
        for _ in range(init_budget):
            x = rng.uniform(-5, 5, dim)
            safe_call(x)
            if calls_made >= budget:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            safe_call(best_x)

        # Phase 2: Nelder-Mead Optimization with Restarts
        # We will run Nelder-Mead for the remaining budget.
        # We'll implement a simple restart if the simplex size shrinks too much.
        
        # Initial simplex around best_x
        # Size of initial simplex
        init_size = 0.5
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = best_x
        for i in range(1, dim + 1):
            simplex[i] = best_x.copy()
            simplex[i, i - 1] += init_size
            # Ensure within bounds
            simplex[i] = np.clip(simplex[i], -5, 5)
        
        # Evaluate initial simplex
        fvals = np.zeros(dim + 1)
        for i in range(dim + 1):
            if calls_made >= budget:
                break
            fvals[i] = safe_call(simplex[i])
        
        # If we ran out of budget during initialization, return
        if calls_made >= budget:
            return best_x, best_fx

        # Nelder-Mead Parameters
        alpha = 1.0    # Reflection
        gamma = 2.0    # Expansion
        rho = 0.5      # Contraction
        sigma = 0.5    # Shrink

        # Main loop
        while calls_made < budget:
            # Sort simplex by fitness
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            # Check for convergence
            # If the best and worst are very close, or simplex size is tiny
            simplex_size = np.linalg.norm(simplex[0] - simplex[-1])
            if simplex_size < 1e-6 or (fvals[-1] - fvals[0]) < 1e-10:
                # Restart: Perturb the best point and create a new simplex
                if calls_made >= budget:
                    break
                
                # Create a new simplex around best_x with a larger step
                # Try to expand the search slightly
                new_simplex = np.zeros((dim + 1, dim))
                new_simplex[0] = best_x
                # Random direction perturbations
                for i in range(1, dim + 1):
                    direction = rng.normal(size=dim)
                    direction /= (np.linalg.norm(direction) + 1e-10)
                    step = 0.5 * simplex_size + 0.1 # Ensure some movement
                    new_simplex[i] = best_x + step * direction
                    new_simplex[i] = np.clip(new_simplex[i], -5, 5)
                
                simplex = new_simplex
                fvals = np.zeros(dim + 1)
                for i in range(dim + 1):
                    if calls_made >= budget:
                        break
                    fvals[i] = safe_call(simplex[i])
                
                if calls_made >= budget:
                    break
                continue

            # Calculate centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # 1. Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = safe_call(xr)
            
            if calls_made >= budget:
                break

            if fvals[0] <= fr < fvals[-2]:
                # Accept reflection
                simplex[-1] = xr
                fvals[-1] = fr
            elif fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = safe_call(xe)
                
                if calls_made >= budget:
                    break

                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                
                xc = np.clip(xc, -5, 5)
                fc = safe_call(xc)
                
                if calls_made >= budget:
                    break

                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    # Move all points except best towards best
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        if calls_made >= budget:
                            break
                        fvals[i] = safe_call(simplex[i])
                    
                    if calls_made >= budget:
                        break

        return best_x, best_fx
