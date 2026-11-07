"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead simplex optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # Initialize simplex
        # Start at origin, which is often a good heuristic for centered functions
        x0 = np.zeros(self.dim)
        
        # Initial step size. Domain is [-5, 5]. 
        # A step of 1.0 allows exploring a reasonable portion of the space.
        delta = 1.0 
        
        # Construct simplex vertices
        # v[0] = x0
        # v[i] = x0 + delta * e_i
        simplex = np.zeros((self.dim + 1, self.dim))
        simplex[0] = x0
        for i in range(1, self.dim + 1):
            simplex[i] = x0
            simplex[i, i - 1] += delta
            
        # Evaluate initial simplex
        fvals = np.array([func(x) for x in simplex])
        
        # Count evaluations used
        evals_used = self.dim + 1
        
        # Nelder-Mead parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrink
        
        # Main loop
        while evals_used < self.budget:
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            x_best = simplex[0]
            x_second = simplex[1]
            x_worst = simplex[-1]
            f_best = fvals[0]
            f_second = fvals[1]
            f_worst = fvals[-1]
            
            # Check for convergence
            if np.abs(f_worst - f_best) < 1e-10:
                break
            
            # Centroid of the best D points
            x_cent = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_ref = x_cent + alpha * (x_cent - x_worst)
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            evals_used += 1
            
            if f_ref < f_best:
                # Try Expansion
                if evals_used < self.budget:
                    x_exp = x_cent + gamma * (x_ref - x_cent)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = func(x_exp)
                    evals_used += 1
                    if f_exp < f_ref:
                        x_new = x_exp
                        f_new = f_exp
                    else:
                        x_new = x_ref
                        f_new = f_ref
                else:
                    x_new = x_ref
                    f_new = f_ref
                
            elif f_ref < f_second:
                # Accept Reflection
                x_new = x_ref
                f_new = f_ref
                
            else:
                # Contraction
                # Check if we have budget for contraction
                if evals_used < self.budget:
                    if f_ref < f_worst:
                        # Inside contraction
                        x_con = x_cent - rho * (x_worst - x_cent)
                        x_con = np.clip(x_con, -5, 5)
                        f_con = func(x_con)
                        evals_used += 1
                        if f_con < f_worst:
                            x_new = x_con
                            f_new = f_con
                        else:
                            # Shrink
                            for i in range(1, self.dim + 1):
                                simplex[i] = x_best + sigma * (simplex[i] - x_best)
                                fvals[i] = func(simplex[i])
                                evals_used += 1
                            continue
                    else:
                        # Outside contraction (f_ref >= f_worst)
                        x_con = x_cent + rho * (x_ref - x_cent)
                        x_con = np.clip(x_con, -5, 5)
                        f_con = func(x_con)
                        evals_used += 1
                        if f_con < f_ref:
                            x_new = x_con
                            f_new = f_con
                        else:
                            # Shrink
                            for i in range(1, self.dim + 1):
                                simplex[i] = x_best + sigma * (simplex[i] - x_best)
                                fvals[i] = func(simplex[i])
                                evals_used += 1
                            continue
                else:
                    # No budget for contraction/shrink, just accept worst or break
                    break
            
            # Replace the worst point with the new point
            simplex[-1] = x_new
            fvals[-1] = f_new
            
        # The return value is ignored, but we must return something
        return x_best, f_best
# EVOLVE-BLOCK-END
