"""Rewritten optimizer for BBOB-style f14 (Different powers) using Nelder-Mead + L-BFGS-B."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Nelder-Mead + L-BFGS-B optimizer with strict budget and time guards."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.start_time = time.time()
        self.time_limit = 290.0  # Safety margin under 330s
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.call_count = 0
        self.start_time = time.time()
        self.best_x = None
        self.best_f = float('inf')
        
        dim = self.dim
        budget = self.budget
        
        def wrapped_func(x):
            self.call_count += 1
            # Time guard
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError("Time limit exceeded")
            # Budget guard
            if self.call_count > budget:
                raise ValueError("Budget exceeded")
            
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
                
            return fx

        # 1. Generate deterministic Halton sequence for initial points
        # Use 8 initial points to cover the space well.
        n_initial = 8
        halton_points = self._halton_sequence(n_initial, dim)
        
        # Evaluate initial points
        initial_results = []
        for p in halton_points:
            try:
                fx = wrapped_func(p)
                initial_results.append((fx, p))
            except (ValueError, TimeoutError):
                break
        
        if self.best_x is None:
            # Fallback if something went wrong immediately
            return np.zeros(dim), float('inf')

        # Sort initial results by fitness
        initial_results.sort(key=lambda x: x[0])
        
        # 2. Run Nelder-Mead from the best initial point, then L-BFGS-B from the result
        # Nelder-Mead is derivative-free and robust for 5D
        # L-BFGS-B refines with gradient information
        
        bounds = [(-5.0, 5.0)] * dim
        
        # Reserve some budget for safety
        reserve = 30
        
        # Try Nelder-Mead from the top 2 initial points
        for i, (fx, x0) in enumerate(initial_results[:2]):
            # Check if we have enough budget
            if self.call_count + 100 > budget - reserve:
                break
                
            if time.time() - self.start_time > self.time_limit - 15:
                break

            try:
                # Run Nelder-Mead
                res_nm = minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                
            except (ValueError, TimeoutError, Exception):
                break

            # Now refine with L-BFGS-B from the Nelder-Mead result
            if self.call_count + 50 > budget - reserve:
                break
                
            if time.time() - self.start_time > self.time_limit - 10:
                break

            try:
                res_lbfgs = minimize(
                    wrapped_func, 
                    res_nm.x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 30, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                
            except (ValueError, TimeoutError, Exception):
                break

        # 3. If we still have budget, do another round of L-BFGS-B from the current best
        if self.call_count + 30 < budget - reserve:
            if time.time() - self.start_time < self.time_limit - 10:
                try:
                    res_final = minimize(
                        wrapped_func, 
                        self.best_x, 
                        method='L-BFGS-B', 
                        bounds=bounds,
                        options={'maxiter': 20, 'ftol': 1e-15, 'gtol': 1e-12}
                    )
                except (ValueError, TimeoutError, Exception):
                    pass

        return self.best_x, self.best_f

    def _halton_sequence(self, n, dim):
        """Generate Halton sequence points in [0, 1)^dim, scaled to [-5, 5]^dim."""
        points = []
        for i in range(1, n + 1):
            point = []
            for j in range(dim):
                base = j + 2  # Start with base 2, 3, 5, ...
                f = 1.0
                r = 0.0
                i_val = i
                while i_val > 0:
                    f /= base
                    r += f * (i_val % base)
                    i_val //= base
                point.append(r)
            # Scale from [0, 1) to [-5, 5]
            point = np.array(point) * 10.0 - 5.0
            points.append(point)
        return points
# EVOLVE-BLOCK-END
