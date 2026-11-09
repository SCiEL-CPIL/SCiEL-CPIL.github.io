import numpy as np
import time
import scipy.optimize


class Optimizer:
    """Multi-start SciPy Optimizer for Rosenbrock-like functions."""

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
        self.deadline = self.start_time + 280.0  # Safety margin before 300s kill

        def eval_func(x):
            """Evaluate with budget/time tracking."""
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")
            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = func(x)
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            return val

        # Starting points: diverse but informed by Rosenbrock structure
        # The true minimum is in [-4, 4]^5.
        # Standard Rosenbrock min is (1, 1, ..., 1).
        # We try points around this and random points.
        rng = np.random.default_rng(42)
        starts = [
            np.ones(self.dim),
            np.zeros(self.dim),
            np.full(self.dim, 2.0),
            np.full(self.dim, 0.5),
            np.full(self.dim, 1.5),
        ]
        # Add random points in the likely region
        for _ in range(5):
            starts.append(rng.uniform(-2, 4, self.dim))
        # Add random points in the full box
        for _ in range(3):
            starts.append(rng.uniform(-5, 5, self.dim))

        unique_starts = []
        for s in starts:
            if not any(np.allclose(s, u, atol=1e-5) for u in unique_starts):
                unique_starts.append(s)

        # We will run Powell and Nelder-Mead.
        # Powell is often very good for Rosenbrock.
        # We split the budget.
        # Let's give 60% to Powell attempts and 40% to Nelder-Mead attempts.
        # Or just run them sequentially until budget runs out.
        
        # Strategy:
        # 1. Run Powell on a few promising starts.
        # 2. Run Nelder-Mead on a few promising starts.
        # 3. If budget remains, run more.

        # To be safe and effective, we'll iterate through starts and methods.
        # We need to be careful with maxfev.
        
        methods = ['Powell', 'Nelder-Mead']
        
        # Shuffle starts to mix up which start gets which method first?
        # Or just assign specific starts to Powell first as it's generally stronger for this.
        
        # Let's try 3 Powell runs and 3 Nelder-Mead runs, each with ~80 calls.
        # Total 480 calls.
        
        num_starts_powell = 3
        num_starts_nm = 3
        calls_per_run = 80
        
        # Ensure we don't exceed budget
        total_planned = (num_starts_powell + num_starts_nm) * calls_per_run
        if total_planned > self.budget:
             calls_per_run = self.budget // (num_starts_powell + num_starts_nm)
             
        # Use the best unique starts
        # Sort starts by distance to (1,1,1,1,1) as a heuristic?
        # Or just take the first few.
        
        # Let's prioritize starts closer to (1,1,1,1,1)
        target = np.ones(self.dim)
        unique_starts.sort(key=lambda x: np.linalg.norm(x - target))
        
        # Select starts for Powell
        powell_starts = unique_starts[:num_starts_powell]
        # Select starts for Nelder-Mead (different ones if possible)
        nm_starts = unique_starts[num_starts_powell:num_starts_powell + num_starts_nm]
        
        # If we don't have enough unique starts, reuse
        if len(nm_starts) < num_starts_nm:
             nm_starts = unique_starts[:num_starts_nm]

        # Run Powell
        for x0 in powell_starts:
            if self.calls_made >= self.budget or time.time() > self.deadline:
                break
            remaining = self.budget - self.calls_made
            if remaining < 10:
                break
            try:
                res = scipy.optimize.minimize(
                    eval_func, 
                    x0, 
                    method='Powell', 
                    options={'maxfev': remaining, 'maxiter': 1000}
                )
            except StopIteration:
                break
            except Exception:
                continue

        # Run Nelder-Mead
        for x0 in nm_starts:
            if self.calls_made >= self.budget or time.time() > self.deadline:
                break
            remaining = self.budget - self.calls_made
            if remaining < 10:
                break
            try:
                res = scipy.optimize.minimize(
                    eval_func, 
                    x0, 
                    method='Nelder-Mead', 
                    options={'maxfev': remaining, 'maxiter': 1000}
                )
            except StopIteration:
                break
            except Exception:
                continue
        
        # If we still have budget, run one more Powell on the best point found
        if self.best_x is not None and self.calls_made < self.budget:
             remaining = self.budget - self.calls_made
             if remaining > 20:
                 try:
                     res = scipy.optimize.minimize(
                         eval_func, 
                         self.best_x, 
                         method='Powell', 
                         options={'maxfev': remaining, 'maxiter': 1000}
                     )
                 except StopIteration:
                     pass
                 except Exception:
                     pass

        # Fallback if no calls were made
        if self.best_x is None:
            x0 = np.ones(self.dim)
            self.best_x = x0
            self.best_f = func(np.clip(x0, -5, 5))

        return self.best_x, self.best_f
