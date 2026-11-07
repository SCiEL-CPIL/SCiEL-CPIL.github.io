"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Multi-start Nelder-Mead optimizer with budget and time guards."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls_made = 0
        self.x_best = None
        self.f_best = float('inf')
        self.start_time = None
        self.deadline = None

    def __call__(self, func):
        self.start_time = time.time()
        # Allow 300s limit, keep 10s margin
        self.deadline = self.start_time + 290.0 
        rng = np.random.default_rng(42)
        
        # Wrapper to track calls and update best
        def wrapped_func(x):
            self.calls_made += 1
            # Clip to box [-5, 5]
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            
            # Update best
            if fx < self.f_best:
                self.f_best = fx
                self.x_best = x_clipped.copy()
            
            return fx

        # Phase 1: Initial Exploration
        # Evaluate a small number of random points to find a good starting basin
        # 5D space, 10 points is cheap and helps avoid starting in a bad region
        n_initial = 10
        for _ in range(n_initial):
            if self.calls_made >= self.budget:
                break
            if time.time() > self.deadline:
                break
            x = rng.uniform(-4.0, 4.0, self.dim)
            wrapped_func(x)

        # Phase 2: Local Optimization
        # Use the best point found so far as the starting point for Nelder-Mead
        # We can also try a few other promising points if budget allows
        # But Nelder-Mead is local, so starting from the global best of phase 1 is best.
        
        # We will run Nelder-Mead in a loop, potentially restarting if it gets stuck
        # or if we want to explore other basins.
        
        # To maximize early progress, we start with the current best.
        # We limit the number of evaluations per local run to ensure we can restart if needed.
        # However, Nelder-Mead on Rosenbrock 5D usually converges in < 100 evals.
        
        # Let's run one main Nelder-Mead optimization from the best initial point.
        # If it finishes quickly and we have budget left, we can try another random start.
        
        # SciPy Nelder-Mead options:
        # maxiter: max function evaluations? No, iterations.
        # We need to be careful with the budget.
        # We will use a custom callback to stop if budget is reached.
        
        def callback(xk):
            if self.calls_made >= self.budget:
                return True
            if time.time() > self.deadline:
                return True
            return False

        # Start from the best point found in Phase 1
        x_start = self.x_best.copy()
        
        # Run Nelder-Mead
        try:
            result = scipy.optimize.minimize(
                wrapped_func, 
                x_start, 
                method='Nelder-Mead',
                options={
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'maxiter': 1000, # High limit, controlled by callback/budget
                    'adaptive': True
                },
                callback=callback
            )
        except Exception:
            pass

        # If we still have budget and time, we might try a few more random restarts
        # to see if we can find a better basin.
        # Given the budget is 500, and Nelder-Mead might use ~100-200, we have room.
        
        # Let's do 2 more restarts if budget allows
        for i in range(2):
            if self.calls_made >= self.budget:
                break
            if time.time() > self.deadline:
                break
            
            # Generate a new random start
            x_new = rng.uniform(-4.0, 4.0, self.dim)
            
            # Quick check: evaluate it
            fx_new = wrapped_func(x_new)
            
            # If it's significantly better than current best, run Nelder-Mead from it
            # Otherwise, maybe just run Nelder-Mead from it to see if it leads to a better minimum
            # Even if fx_new is worse, the local minimum might be better.
            
            # We need to be careful not to waste budget on a bad start.
            # But Nelder-Mead is relatively cheap.
            
            # We'll run Nelder-Mead from x_new
            try:
                result2 = scipy.optimize.minimize(
                    wrapped_func, 
                    x_new, 
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'maxiter': 500,
                        'adaptive': True
                    },
                    callback=callback
                )
            except Exception:
                pass

        return self.x_best, self.f_best
# EVOLVE-BLOCK-END
