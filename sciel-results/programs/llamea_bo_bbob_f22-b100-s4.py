import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 290  # 10s safety margin

    def __call__(self, func):
        # Wrapper to count calls and enforce deadline
        def wrapped_func(x):
            if self.calls >= self.budget:
                return self.best_f
            if time.time() > self.deadline:
                return self.best_f
            x = np.asarray(x)
            # Clip to box
            x = np.clip(x, -5, 5)
            val = func(x)
            self.calls += 1
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val
        
        # 1. Initial diverse sampling
        # Use a quasi-random sequence for better coverage
        rng = np.random.default_rng(42)
        
        # Generate a few initial points
        n_init = 10
        # Halton sequence or just random? Random is fine with 10 points in 5D.
        # Let's use a mix: center + randoms
        points = []
        points.append(np.zeros(self.dim))
        for _ in range(n_init - 1):
            points.append(rng.uniform(-5, 5, self.dim))
            
        # Evaluate initial points
        for p in points:
            if self.calls >= self.budget or time.time() > self.deadline:
                break
            wrapped_func(p)
        
        # Sort initial points by fitness (we need to store them)
        # Since wrapped_func updates self.best, we need to track individual fitness for sorting starts
        # Let's re-evaluate or just track during the loop above?
        # Better: store (x, f) pairs
        
        candidates = []
        for p in points:
            if self.calls >= self.budget or time.time() > self.deadline:
                break
            # We already evaluated in the loop above? 
            # No, the loop above called wrapped_func. 
            # We need the fitness values to sort.
            # Let's refactor: evaluate and store.
            pass
        
        # Refactored Init:
        candidates = []
        for p in points:
            if self.calls >= self.budget or time.time() > self.deadline:
                break
            f_val = wrapped_func(p)
            candidates.append((f_val, p))
            
        # Sort by fitness
        candidates.sort(key=lambda x: x[0])
        
        # 2. Local Optimization from best candidates
        # Try to optimize from the best 3 candidates
        for f_val, p in candidates[:3]:
            if self.calls >= self.budget or time.time() > self.deadline:
                break
            
            # Run L-BFGS-B
            # maxiter is not the call limit, so we rely on the wrapper's call limit
            # However, minimize might run away if we don't stop it.
            # The wrapper returns best_f if limit reached, which might confuse the optimizer.
            # It's safer to check the budget before starting a new minimize block.
            
            try:
                res = minimize(
                    wrapped_func, 
                    p, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                # Update best if res.fun is better
                # wrapped_func already updated self.best_f
            except Exception:
                pass
                
        return self.best_x, self.best_f
