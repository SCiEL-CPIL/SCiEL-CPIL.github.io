"""Combined Multi-Start Powell + (1+1)-ES Optimizer."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Multi-Start Powell's Method with (1+1)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        # Use 280s limit to be safe within 300s
        self.deadline = self.start_time + 280.0

        def wrapped_func(x):
            # Check limits
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")

            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = func(x)
            
            # Update global best
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        def es_refine(x_center, remaining_budget):
            """(1+1)-ES refinement around x_center using remaining budget."""
            if remaining_budget < 10:
                return
            if time.time() > self.deadline:
                return
            
            rng = np.random.default_rng(42)
            x = x_center.copy()
            fx = self.best_f
            sigma = 1.0  # Start with moderate sigma for local refinement
            
            for _ in range(remaining_budget):
                if self.calls_made >= self.budget:
                    break
                if time.time() > self.deadline:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = wrapped_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2  # Slightly more aggressive growth
                else:
                    sigma *= 0.85  # Decay sigma on failure
                    
                # Stop if sigma becomes too small
                if sigma < 1e-6:
                    break

        # Multi-start configuration
        # Use diverse starting points
        rng = np.random.default_rng(12345)
        
        starts = []
        starts.append(np.zeros(self.dim))
        starts.append(rng.uniform(-5, 5, self.dim))
        starts.append(rng.uniform(-5, 5, self.dim))
        
        n_starts = len(starts)
        
        # Phase 1: Multi-start Powell's Method
        # Distribute budget for Powell, keeping ~30% for ES refinement
        powell_budget = int(self.budget * 0.7)
        es_budget = self.budget - powell_budget
        budget_per_start = powell_budget // n_starts

        for i, x0 in enumerate(starts):
            if self.calls_made >= powell_budget:
                break
            if time.time() > self.deadline:
                break

            # Define callback to stop SciPy if we hit limits
            def callback(xk):
                return self.calls_made >= powell_budget or time.time() > self.deadline

            try:
                # Use Powell's Method
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='Powell',
                    options={
                        'maxiter': 30,  # Limit iterations to prevent stalling
                    },
                    callback=callback
                )
                
                # Update best if result is valid
                if res.fun < self.best_f:
                    self.best_f = res.fun
                    self.best_x = np.clip(res.x, -5, 5)

            except StopIteration:
                break
            except Exception:
                # If an error occurs, try next start if budget allows
                continue

        # Phase 2: (1+1)-ES refinement around the best point found
        if self.best_x is not None and self.calls_made < self.budget:
            remaining = self.budget - self.calls_made
            es_refine(self.best_x, remaining)

        # If no calls were made (shouldn't happen), return initial guess
        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_f = func(np.clip(self.best_x, -5, 5))

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
