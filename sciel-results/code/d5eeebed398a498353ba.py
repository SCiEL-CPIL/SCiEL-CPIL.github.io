import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0
        self.start_time = time.time()
        # Evaluator kills after 330s. We should aim for ~300s max.
        self.time_limit = 280.0

    def __call__(self, func):
        self.call_count = 0
        self.start_time = time.time()
        self.best_x = None
        self.best_f = float('inf')
        
        def wrapped_func(x):
            self.call_count += 1
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError
            x = np.clip(x, -5.0, 5.0)
            return func(x)

        def update_best(x, f):
            if f < self.best_f:
                self.best_f = f
                self.best_x = np.clip(x, -5.0, 5.0).copy()

        n = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0) for _ in range(n)]

        # Phase 1: Diverse Initial Sampling
        # Use a Halton sequence for low-discrepancy coverage
        # 20 points should be sufficient for d=5 to cover the space well
        n_starts = 10
        
        def halton_sequence(n_points, dim):
            # Generate Halton sequence
            def halton(index, base):
                result = 0
                f = 1
                i = index
                while i > 0:
                    f /= base
                    result += f * (i % base)
                    i //= base
                return result
            
            # Use first n_points indices (1-based for Halton)
            indices = np.arange(1, n_points + 1)
            bases = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])[:dim]
            points = np.zeros((n_points, dim))
            for d in range(dim):
                for i in range(n_points):
                    points[i, d] = halton(indices[i], int(bases[d]))
            # Scale to [-5, 5]
            points = points * 10.0 - 5.0
            return points

        X0 = halton_sequence(n_starts, n)
        
        # Evaluate initial points
        candidates = []
        for x in X0:
            if self.call_count >= budget or time.time() - self.start_time > self.time_limit:
                break
            try:
                f = wrapped_func(x)
                candidates.append((f, x.copy()))
                update_best(x, f)
            except Exception:
                pass

        if not candidates:
            return self.best_x, self.best_f

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Local Optimization
        remaining = budget - self.call_count
        
        # We will try to polish the top 5 candidates
        # Allocate remaining budget among them
        # L-BFGS-B is cheap per iteration but might need many evaluations
        # Let's give each candidate ~ remaining / 5 calls, but with a cap
        # For d=5, L-BFGS-B usually converges in < 50-100 evaluations if close
        
        top_k = min(3, len(candidates))
        if top_k > 0:
            calls_per_opt = max(20, remaining // top_k)
            
            for i in range(top_k):
                if self.call_count >= budget or time.time() - self.start_time > self.time_limit:
                    break
                
                f0, x0 = candidates[i]
                
                try:
                    res = minimize(wrapped_func, x0, method='L-BFGS-B', bounds=bounds,
                                   options={'maxfun': calls_per_opt, 'maxiter': 150})
                    if res.fun < self.best_f:
                        self.best_f = res.fun
                        self.best_x = np.clip(res.x, -5.0, 5.0).copy()
                except Exception:
                    pass

        # Phase 3: Final Polish on the absolute best
        if self.call_count < budget - 10 and self.best_x is not None:
            remaining_final = budget - self.call_count - 5
            if remaining_final > 10:
                try:
                    res = minimize(wrapped_func, self.best_x, method='L-BFGS-B', bounds=bounds,
                                   options={'maxfun': remaining_final, 'maxiter': 300})
                    if res.fun < self.best_f:
                        self.best_f = res.fun
                        self.best_x = np.clip(res.x, -5.0, 5.0).copy()
                except Exception:
                    pass

        return self.best_x, self.best_f
