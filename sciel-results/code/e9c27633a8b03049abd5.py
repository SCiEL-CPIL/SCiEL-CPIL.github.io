import numpy as np
import time
import sys

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._start_time = None
        self._deadline = None
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        self._start_time = time.time()
        # 330s limit, keep 5s margin
        self._deadline = self._start_time + 325
        self._call_count = 0
        self.budget_remaining = self.budget
        self._best_x = np.zeros(self.dim)
        self._best_f = np.inf
        self._rng = np.random.default_rng(42)

        def safe_call(x):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                return 1e10
            self._call_count += 1
            self.budget_remaining -= 1
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        # Strategy: Interleave Sobol global exploration with Nelder-Mead local refinement
        # This provides steady early progress (good for AOCC) and robust final convergence.

        dim = self.dim
        budget = self.budget

        # Generate Sobol sequence for global coverage
        # We want to spread out the global evaluations
        # Let's plan for about 30-40 global points and refine the best few
        # Or interleave: evaluate a Sobol point, if it's good, refine it.
        
        # A better strategy for AOCC:
        # 1. Start with a few random/Sobol points to get a quick sense of the landscape.
        # 2. Identify the best point.
        # 3. Refine it with Nelder-Mead.
        # 4. Generate more Sobol points, evaluate them.
        # 5. If a new Sobol point is better than the current best, refine it.
        # 6. Repeat until budget is exhausted.
        # 7. Final polish on the global best.

        from scipy.optimize import minimize
        from scipy.stats import qmc

        # Sobol sequence
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        # Generate a large number of points, we'll use them on the fly
        # 2^10 = 1024 points is plenty for 500 budget
        sobol_points = sobol.random(1024) * 10.0 - 5.0  # Map to [-5, 5]

        # Phase 1: Initial Exploration
        # Evaluate first 5 Sobol points to get a quick baseline
        for i in range(5):
            x = sobol_points[i]
            safe_call(x)

        # Phase 2: Interleaved Exploration and Exploitation
        # We will iterate through the Sobol points.
        # After each new point, if it's the new best, we do a quick local refinement.
        # If it's not the new best, we just move on to the next Sobol point.
        
        current_best_f = self._best_f
        current_best_x = self._best_x.copy()
        
        sobol_idx = 5  # Start from index 5
        
        # We'll use a loop that checks budget and time
        while self._call_count < budget - 50 and time.time() < self._deadline:
            # Get next Sobol point
            if sobol_idx >= len(sobol_points):
                break
            
            x_new = sobol_points[sobol_idx]
            sobol_idx += 1
            
            f_new = safe_call(x_new)
            
            # If this new point is better than the current best, refine it
            if f_new < current_best_f:
                current_best_f = f_new
                current_best_x = x_new.copy()
                
                # Quick local refinement (Nelder-Mead)
                # Use a small budget for this refinement to allow more global exploration
                # maxiter=30 is usually enough for a quick improvement
                try:
                    res = minimize(
                        safe_call,
                        current_best_x,
                        method='Nelder-Mead',
                        options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
                    )
                    if res.fun < 1e9:
                        current_best_f = res.fun
                        current_best_x = res.x.copy()
                except Exception:
                    pass
                
            # If we have spent a lot of budget, we might want to stop exploring
            # and focus on refining the best point found so far.
            # Let's reserve the last 50 calls for a final deep refinement.

        # Phase 3: Final Polish
        # Use the remaining budget to deeply refine the best point found.
        if self._call_count < budget and time.time() < self._deadline:
            try:
                # Use a larger maxiter for the final polish
                # The adaptive Nelder-Mead will stop early if it converges
                res = minimize(
                    safe_call,
                    self._best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                )
            except Exception:
                pass

        return self._best_x, self._best_f
