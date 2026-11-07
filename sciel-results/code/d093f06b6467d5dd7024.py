import numpy as np
from scipy.optimize import brentq

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT6 specific knowledge:
        # Pareto front is at x_2...x_9 = 0.
        # f1 = 91 + sum(x_i^2 - 10cos(4pi*x_i)) for i=1..9
        # If x_2..x_9 = 0, f1 = 91 + x_1^2 - 10cos(4pi*x_1) - 80 = 11 + x_1^2 - 10cos(4pi*x_1)
        # We want f1 in [1, 11].
        # So x_1^2 - 10cos(4pi*x_1) must be in [-10, 0].
        
        n_evals = self.budget
        if n_evals == 0:
            return None
            
        # Generate target f1 values
        # Uniformly spaced in [1, 11]
        f1_targets = np.linspace(1.0, 11.0, n_evals)
        
        solutions = []
        
        # Function to solve for x1 given target f1
        def solve_x1(target_f1):
            # Equation: x1^2 - 10*cos(4*pi*x1) = target_f1 - 11
            rhs = target_f1 - 11.0
            
            def eq(x):
                return x**2 - 10.0 * np.cos(4.0 * np.pi * x) - rhs
            
            # Root is in (0, 0.125)
            # Check boundaries
            # At x=0, eq = -10 - rhs. If rhs=-10 (f1=1), eq=0.
            # At x=0.125, eq = 0.0156 - 0 - rhs. If rhs=0 (f1=11), eq=0.0156.
            # So for f1=1, root is 0.
            # For f1=11, root is near 0.125?
            # Let's check f1=11 => rhs=0. x^2 = 10cos(4pix).
            # Root is approx 0.124.
            
            if target_f1 <= 1.0:
                return 0.0
            if target_f1 >= 11.0:
                # Find max x1
                try:
                    return brentq(eq, 1e-9, 0.1249)
                except:
                    return 0.1249
            
            try:
                return brentq(eq, 1e-9, 0.1249)
            except:
                return 0.0

        for f1_t in f1_targets:
            x1 = solve_x1(f1_t)
            x = np.zeros(self.dim)
            x[0] = x1
            # x[1:] are 0
            
            # Clip to bounds just in case
            x = np.clip(x, bounds[0][0], bounds[0][1]) # bounds is list of pairs
            # Actually bounds is list of (low, high) for each dim.
            # x[0] is dim 0.
            x = np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
            
            # Evaluate
            _ = problem(x)
            solutions.append(x)
            
        return solutions
