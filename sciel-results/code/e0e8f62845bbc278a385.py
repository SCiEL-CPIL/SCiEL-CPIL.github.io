import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        call_count = 0
        best_x = None
        best_f = np.inf
        candidates = [] # List of (f_val, x)

        def safe_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget or time.time() > deadline - 2:
                raise ValueError("Budget or time exceeded")
            call_count += 1
            val = float(func(x))
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float)
            candidates.append((val, np.array(x, dtype=float)))
            return val

        # Phase 1: Initial Sampling
        # We use a mix of random and structured points to ensure coverage.
        # Rosenbrock minimum is at (1, 1, ..., 1) in standard form.
        # With hidden shift/rotation, we don't know where it is, but it's in [-4, 4]^5.
        # We sample densely in the center.
        
        n_random = 120
        # Sample in [-3, 3] to focus on the likely region, with a few in [-5, 5]
        random_pts = rng.uniform(-3, 3, (n_random, self.dim))
        
        # Add some structured points around the likely center (0,0,0) and (1,1,1) etc.
        # Since we don't know the shift, we rely on random density.
        # But adding the standard Rosenbrock center (1,1,1,1,1) and its negations might help 
        # if the rotation is small or aligned.
        structured = [
            np.zeros(self.dim),
            np.ones(self.dim), # Standard min
            np.ones(self.dim) * -1.0,
            np.ones(self.dim) * 0.5,
            np.ones(self.dim) * -0.5,
            np.ones(self.dim) * 2.0,
            np.ones(self.dim) * -2.0,
        ]
        
        all_pts = np.vstack([structured, random_pts])
        
        # Evaluate all initial points
        for p in all_pts:
            if call_count >= self.budget or time.time() > deadline - 10:
                break
            try:
                safe_func(p)
            except ValueError:
                break

        if best_x is None:
            return np.zeros(self.dim), 0.0

        # Sort candidates by function value
        candidates.sort(key=lambda item: item[0])
        
        # Phase 2: Multi-Start L-BFGS-B
        # We pick the top K candidates that are sufficiently distinct.
        # Distinctness check: Euclidean distance > 0.5
        selected_starts = []
        for f_val, x in candidates:
            if len(selected_starts) >= 5:
                break
            is_distinct = True
            for _, sx in selected_starts:
                if np.linalg.norm(x - sx) < 0.5:
                    is_distinct = False
                    break
            if is_distinct:
                selected_starts.append((f_val, x))

        for i, (f_val, start_x) in enumerate(selected_starts):
            remaining = self.budget - call_count
            if remaining < 20 or time.time() > deadline - 5:
                break
            
            # Allocate budget based on rank. Top start gets more budget.
            # But L-BFGS-B is cheap. We just set maxfun to remaining.
            try:
                # Use L-BFGS-B. It handles bounds well.
                # We use a tight tolerance for the final polish later, so here we just need to get into the basin.
                res = minimize(
                    safe_func,
                    start_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 200, 
                        'maxfun': min(remaining, 100), # Limit per start to ensure we finish
                        'maxls': 30, 
                        'gtol': 1e-6, 
                        'ftol': 1e-10
                    }
                )
                # The safe_func updates best_f and best_x automatically if it finds something better.
                # But we should check if res.x is better than current best (in case safe_func didn't trigger update due to some edge case, though it should).
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.array(res.x, dtype=float)
            except Exception:
                pass

        # Phase 3: Final High-Precision Polish
        # If we have time and budget, do one more L-BFGS-B with very tight tolerances on the best point found.
        remaining = self.budget - call_count
        if remaining > 10 and time.time() < deadline - 2 and best_x is not None:
            try:
                res_polish = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 500,
                        'maxfun': remaining,
                        'maxls': 50,
                        'gtol': 1e-12,
                        'ftol': 1e-15
                    }
                )
                if res_polish.fun < best_f:
                    best_f = float(res_polish.fun)
                    best_x = np.array(res_polish.x, dtype=float)
            except Exception:
                pass

        return best_x, best_f
