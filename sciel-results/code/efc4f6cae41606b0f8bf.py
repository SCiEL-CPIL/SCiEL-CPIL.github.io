"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit
        
        # Determine number of restarts
        # We want enough restarts to cover the space, but enough budget per restart to converge.
        # 5D is low. 20 restarts is good.
        n_restarts = 20
        if self.budget < 100:
            n_restarts = max(5, self.budget // 10)
            
        # Ensure we don't ask for more initial points than budget allows
        n_restarts = min(n_restarts, self.budget - 5) 
        if n_restarts < 1:
            n_restarts = 1

        rng = np.random.default_rng(42)
        
        # Generate initial points
        initial_points = [rng.uniform(-5, 5, self.dim) for _ in range(n_restarts)]
        
        # Evaluate initial points
        initial_values = []
        for x in initial_points:
            initial_values.append(func(x))
            
        # Check deadline after initial evals
        if time.time() > deadline:
            best_idx = np.argmin(initial_values)
            return initial_points[best_idx], initial_values[best_idx]

        # Sort points by value
        order = np.argsort(initial_values)
        
        best_x = initial_points[order[0]]
        best_f = initial_values[order[0]]
        
        # Budget allocation
        remaining_budget = self.budget - n_restarts
        budget_per_restart = remaining_budget // n_restarts
        
        # If budget_per_restart is too small, NM won't work well. 
        # But 20 restarts with 24 calls each is okay.
        
        for idx in order:
            if time.time() > deadline:
                break
            
            x_start = initial_points[idx].copy()
            f_start = initial_values[idx]
            
            # Run Nelder-Mead
            x_opt, f_opt = self._nelder_mead(func, x_start, f_start, budget_per_restart, deadline)
            
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, max_iter, deadline):
        # Nelder-Mead Simplex
        dim = self.dim
        # Initialize simplex
        # x0 is the best point.
        # Create simplex: x0, and x0 + delta * e_i
        delta = 0.1
        simplex = [x0.copy()]
        for i in range(dim):
            xi = x0.copy()
            xi[i] += delta
            simplex.append(xi)
        
        # Evaluate simplex
        fvals = [f0]
        for i in range(1, dim + 1):
            fvals.append(func(simplex[i]))
            
        # Check deadline
        if time.time() > deadline:
            return x0, f0

        # Parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Main loop
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            # Sort simplex by function value
            indices = np.argsort(fvals)
            simplex = [simplex[i] for i in indices]
            fvals = [fvals[i] for i in indices]
            
            # Stop condition: simplex is small
            # Calculate spread
            spread = np.max([np.linalg.norm(simplex[0] - simplex[i]) for i in range(1, dim+1)])
            if spread < 1e-6:
                break
                
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            
            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - simplex[-1])
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                # Outside contraction if f_r < fvals[-1], else inside
                if f_r < fvals[-1]:
                    x_c = centroid + 0.5 * (x_r - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                else:
                    x_c = centroid - 0.5 * (centroid - simplex[-1])
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    if f_c <= fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])

        # Return best point in simplex
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]
# EVOLVE-BLOCK-END
