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
        self.start_time = time.time()
        self.time_limit = 250  # Safety margin below 300s

    def __call__(self, func):
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.start_time = time.time()

        # Wrapper to count calls and enforce budget/time
        def wrapped_func(x):
            self.calls += 1
            if self.calls > self.budget:
                raise RuntimeError("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise RuntimeError("Time limit exceeded")
            # Clip to box
            x = np.clip(x, -5, 5)
            return func(x)

        # Generate diverse starting points
        # The optimum is in [-4, 4]^5.
        # Start with center and some random points.
        rng = np.random.default_rng(42)
        
        starts = []
        starts.append(np.zeros(self.dim))
        
        # Add random points in [-4, 4]
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, self.dim))
            
        # Add a few points near the boundaries of the feasible region for the optimum
        # Just in case the "attractive sector" is skewed
        starts.append(np.full(self.dim, 3.0))
        starts.append(np.full(self.dim, -3.0))

        # Strategy:
        # 1. Run Nelder-Mead from each start point.
        # 2. Limit evaluations per run to ensure we have budget for all.
        # 500 budget / 6 starts ~ 80 calls per run.
        # Nelder-Mead in 5D uses ~5-10 calls/iter. 80 calls is ~10-15 iters.
        # This might not be enough for full convergence if the landscape is tricky.
        # Better approach:
        # Run one NM with a loose tolerance and moderate budget.
        # Then take the best and run another NM with tight tolerance and remaining budget.
        
        # Phase 1: Exploration
        # Let's allocate 70% of budget to exploration, 30% to exploitation.
        # But NM is fast. Let's just run NM on all starts with a cap.
        
        # We will run NM sequentially.
        # To avoid one bad start eating all budget, we cap calls per run.
        # Let's say max 100 calls per start.
        
        max_calls_per_run = 100
        
        for x0 in starts:
            if self.calls >= self.budget:
                break
            if time.time() - self.start_time > self.time_limit:
                break
                
            # Calculate remaining budget for this run
            remaining = self.budget - self.calls
            if remaining <= 0:
                break
            
            # We want to use at most max_calls_per_run or remaining
            # But scipy doesn't have a strict "max_fun" for all methods in all versions?
            # Actually, `maxiter` is the control. We can estimate.
            # Or we can just let it run and catch the exception from the wrapper.
            # However, if it throws an exception, we lose the result of that run?
            # No, we update best_x/best_f inside the wrapper? No, the wrapper just returns f.
            # We need to capture the result of minimize.
            
            # To be safe and capture intermediate bests, we can't easily hook into NM's internal steps
            # without modifying the wrapper to update self.best_x if f < self.best_f.
            
            # Let's modify wrapped_func to update bests.
            # But wrapped_func is defined above. Let's redefine or use a closure that updates self.
            
            # Re-defining wrapped_func to update self.best
            # (Already done in the definition above? No, I need to add the update logic)
            
            # Let's restart the logic inside __call__ with a proper wrapper.
            
        # I will rewrite the __call__ logic properly below.
        
        # --- Corrected Logic ---
        
        def eval_func(x):
            self.calls += 1
            if self.calls > self.budget:
                # Return a large value to signal stop, but raise is better for strict budget?
                # The task says "More than 500 calls ... makes the whole result invalid".
                # So we must NOT exceed 500.
                # If we are at 500, we should not call again.
                # If we call the 501st, it's invalid.
                # So we should check BEFORE calling.
                pass # Handled outside
        
        # Let's use a counter and check before calling func in the wrapper.
        
        def strict_func(x):
            if self.calls >= self.budget:
                # This should not happen if we manage budget externally, 
                # but if NM calls it again, we must prevent the 501st call.
                # Returning a large number might confuse the optimizer but prevents invalid call.
                # However, the task says "More than 500 calls... invalid".
                # So if we call func 501 times, we fail.
                # We must ensure the 501st call to `func` (the original) never happens.
                # So we check in the wrapper.
                return 1e10 # Large penalty, but doesn't count as a call to original func?
                # Wait, the wrapper IS the function passed to minimize.
                # If the wrapper returns 1e10, it doesn't call `func`.
                # So `self.calls` doesn't increment?
                # Let's be precise.
                pass

        # Proper Wrapper
        def wrapper(x):
            if self.calls >= self.budget:
                # Do not call func. Return a bad value to stop optimization.
                return 1e10
            self.calls += 1
            if time.time() - self.start_time > self.time_limit:
                 return 1e10
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            
            # Update best
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x_clipped.copy()
                
            return f_val

        # Now run Nelder-Mead from starts
        
        # Phase 1: Coarse search
        # Use a moderate maxiter.
        # In 5D, NM takes about 5-10 function calls per iteration.
        # Let's allow 50 iterations per start. ~250-500 calls.
        # We have 6 starts. We can't do 50 iters for all.
        # Let's do 10 iterations per start first. ~50-100 calls per start.
        # 6 * 100 = 600 > 500.
        # So let's do 4 starts, 10 iters each. ~400 calls.
        
        # Actually, let's just run one start with a high budget, and if it doesn't converge well, try others?
        # No, the landscape might have multiple basins.
        
        # Strategy:
        # 1. Try Center (0,0,0,0,0) with 200 budget (approx 20-40 iters).
        # 2. If budget remains, try 3 random points with 50 budget each.
        # 3. Polish best with remaining budget.
        
        # Let's implement a helper to run NM with a specific call limit.
        # Since scipy doesn't have a direct "max_fun", we rely on the wrapper returning 1e10 
        # and NM converging/stopping, or we just let it run until maxiter.
        # But if it hits maxiter, it stops.
        # If we set maxiter=100, it might use ~500 calls.
        
        # Let's try:
        # Start 1: Center. maxiter=50.
        # Start 2: Random. maxiter=20.
        # Start 3: Random. maxiter=20.
        # Start 4: Random. maxiter=20.
        
        # And update bests in wrapper.
        
        starts = [
            np.zeros(self.dim),
            rng.uniform(-4, 4, self.dim),
            rng.uniform(-4, 4, self.dim),
            rng.uniform(-4, 4, self.dim),
            np.full(self.dim, 2.5),
            np.full(self.dim, -2.5)
        ]
        
        # Allocate budget roughly
        # 1st: 200 calls
        # Others: 50 calls each
        # Total: 200 + 5*50 = 450. Leaves 50 for polish.
        
        # We can't strictly control calls per run easily without a custom counter in the wrapper 
        # that resets or limits per "run" context.
        # But we can just run them sequentially and let the global budget cap it.
        
        # To ensure we don't spend all budget on the first one, we can set maxiter low.
        # 50 iters * 5 calls/iter = 250 calls.
        
        for i, x0 in enumerate(starts):
            if self.calls >= self.budget - 10: # Reserve 10 for safety
                break
            if time.time() - self.start_time > self.time_limit:
                break
                
            # Set maxiter based on remaining budget and priority
            # First start gets more, others less.
            if i == 0:
                max_iter = 40
            else:
                max_iter = 10
                
            try:
                res = minimize(
                    wrapper, 
                    x0, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-4, # Relaxed tolerance for exploration
                        'fatol': 1e-4
                    }
                )
            except Exception:
                pass

        # Phase 2: Polish
        # If we have a best_x, run NM from there with tight tolerance and remaining budget.
        if self.best_x is not None and self.calls < self.budget - 5:
            try:
                res = minimize(
                    wrapper,
                    self.best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100, # High, but will stop when budget hits via wrapper penalty
                        'xatol': 1e-8, # Tight
                        'fatol': 1e-8
                    }
                )
            except Exception:
                pass

        return self.best_x, self.best_f

    def __call__(self, func):
        # Re-implementing cleanly to avoid the draft mess above
        
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.start_time = time.time()
        
        def wrapper(x):
            if self.calls >= self.budget:
                return 1e10
            if time.time() - self.start_time > 250:
                return 1e10
                
            self.calls += 1
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x_clipped.copy()
                
            return f_val

        rng = np.random.default_rng(123)
        
        # Start points
        starts = [
            np.zeros(self.dim),
            rng.uniform(-4, 4, self.dim),
            rng.uniform(-4, 4, self.dim),
            rng.uniform(-4, 4, self.dim),
            np.full(self.dim, 3.0),
            np.full(self.dim, -3.0)
        ]
        
        # Phase 1: Exploration
        # We want to try multiple basins.
        # Limit iterations to prevent one run from eating all budget.
        
        for i, x0 in enumerate(starts):
            if self.calls >= self.budget - 20:
                break
            if time.time() - self.start_time > 250:
                break
                
            # Assign budget shares
            # First point: ~200 calls
            # Next 4: ~50 calls each
            # Last: ~30 calls
            if i == 0:
                max_iter = 30
            elif i < 5:
                max_iter = 10
            else:
                max_iter = 5
                
            try:
                minimize(
                    wrapper, 
                    x0, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-3,
                        'fatol': 1e-3
                    }
                )
            except Exception:
                pass

        # Phase 2: Polish
        # Use remaining budget to refine the best point found.
        if self.best_x is not None and self.calls < self.budget - 5:
            try:
                minimize(
                    wrapper,
                    self.best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200, # Will stop early if budget runs out or converged
                        'xatol': 1e-8,
                        'fatol': 1e-8
                    }
                )
            except Exception:
                pass

        return self.best_x, self.best_f
