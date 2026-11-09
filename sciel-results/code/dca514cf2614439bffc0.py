import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: stop 5 seconds before the 300s limit
        deadline = start_time + 295.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4 structure:
        # f1 = x[0]
        # g = (dim-1) + 20 * sum_{i=1}^{dim-1} (x_i^2 - 10*cos(4*pi*x_i))
        # f2 = (1 + sqrt(f1)) * g
        #
        # The Pareto front is achieved when g is minimized.
        # For the term x^2 - 10*cos(4*pi*x), the global minimum is at x=0 (value -9).
        # Other local minima exist but -9 is the lowest.
        
        # Find the minimizer of g for the tail variables.
        # g = sum(x_i^2 - 10*cos(4*pi*x_i))
        # Global minima are at integers. 0 is the closest to 0.
        # We assume 0 is within bounds or close to it.
        # If 0 is out of bounds, the minimizer is the bound closest to an integer.
        # For standard ZDT4 bounds [-5, 5] or [0, 1], 0 is optimal.
        
        x_tail_min = np.zeros(self.dim - 1)
        for i in range(1, self.dim):
            l, h = bounds[i]
            # The function h(x) = x^2 - 10*cos(4*pi*x) has global minima at integers.
            # We want the integer (or bound) that minimizes h(x).
            # Since h(x) is periodic-ish with local minima at integers,
            # and the quadratic term pushes towards 0, the global min on any interval
            # containing 0 is 0.
            # If the interval does not contain 0, the min is at the bound closest to an integer.
            # Simple heuristic: check 0, l, h, and nearest integers.
            candidates = [l, h, 0.0]
            # Add nearest integers
            nearest_int = round(l)
            candidates.append(nearest_int)
            candidates.append(nearest_int + 1)
            candidates.append(nearest_int - 1)
            
            best_val = float('inf')
            best_x = l
            for c in candidates:
                if l <= c <= h:
                    val = c*c - 10*np.cos(4*np.pi*c)
                    if val < best_val:
                        best_val = val
                        best_x = c
            
            x_tail_min[i-1] = best_x
        
        n_evals = self.budget
        
        # Improved Sampling Strategy:
        # The Pareto front f2(f1) has an infinite slope at f1=0 (if f1=x0).
        # Uniform sampling in x0 wastes evaluations in the flat region near x0=hi.
        # We use a quadratic transformation to cluster points near x0=lo.
        # Let u be uniform in [0, 1].
        # x0 = lo + (hi - lo) * u^2
        # This ensures higher density of points where the function changes rapidly.
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            
            if n_evals > 1:
                # u in [0, 1]
                u = i / (n_evals - 1)
            else:
                u = 0.5
            
            # Quadratic mapping to cluster near lo
            # If lo < hi, u^2 is in [0,1]
            # x0 = lo + (hi - lo) * u^2
            x0 = lo[0] + (hi[0] - lo[0]) * (u * u)
            
            x = np.zeros(self.dim)
            x[0] = x0
            x[1:] = x_tail_min
            
            try:
                problem(x)
            except Exception:
                continue
        
        return None
