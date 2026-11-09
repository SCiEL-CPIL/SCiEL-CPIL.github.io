"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Nelder-Mead for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        dim = self.dim
        budget = self.budget

        # Nelder-Mead parameters
        x0 = np.zeros(dim)
        
        # Max iterations based on budget
        # Each iteration uses dim + 1 evaluations (simplex update)
        # Plus the initial simplex evaluation (dim + 1)
        # Total calls approx (iterations + 1) * (dim + 1)
        # We want to stay under budget.
        # Let's calculate max_iter conservatively.
        max_iter = int(budget / (dim + 1)) - 2
        
        # Initial simplex
        # Start at 0, step size 1.0
        delta = 1.0
        simplex = [x0]
        for i in range(dim):
            xi = x0.copy()
            xi[i] += delta
            simplex.append(xi)
            
        options = {
            'maxiter': max_iter,
            'xatol': 1e-12,
            'fatol': 1e-12,
            'adaptive': True,
            'initial_simplex': simplex
        }
        
        call_count = 0
        best_x = x0.copy()
        best_f = float('inf')
        
        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if time.time() > deadline:
                raise RuntimeError("Timeout")
            if call_count >= budget:
                raise RuntimeError("Budget Exceeded")
            
            # Clip to box
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            call_count += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        try:
            minimize(wrapped_func, x0, method='Nelder-Mead', options=options)
        except Exception:
            pass

        return best_x, best_f
# EVOLVE-BLOCK-END
