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
        # 20 points for d=5 to cover the space well
        n_starts = 20
        
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
        
        # Phase 2: Local Optimization with coordinate descent pre-polish
        remaining = budget - self.call_count
        
        # We will try to polish the top 5 candidates
        # Allocate remaining budget among them
        top_k = min(5, len(candidates))
        if top_k > 0:
            calls_per_opt = max(20, remaining // top_k)
            
            for i in range(top_k):
                if self.call_count >= budget or time.time() - self.start_time > self.time_limit:
                    break
                
                f0, x0 = candidates[i]
                
                # Coordinate descent pre-polish to get closer to local minimum
                x_cd = x0.copy()
                f_cd = f0
                step_sizes = [2.0, 1.0, 0.5, 0.25, 0.1]
                for step in step_sizes:
                    if self.call_count >= budget or time.time() - self.start_time > self.time_limit:
                        break
                    for dim_idx in range(n):
                        if self.call_count >= budget or time.time() - self.start_time > self.time_limit:
                            break
                        # Try positive and negative steps
                        for direction in [1.0, -1.0]:
                            if self.call_count >= budget or time.time() - self.start_time > self.time_limit:
                                break
                            x_try = x_cd.copy()
                            x_try[dim_idx] += direction * step
                            x_try = np.clip(x_try, -5.0, 5.0)
                            try:
                                f_try = wrapped_func(x_try)
                                if f_try < f_cd:
                                    f_cd = f_try
                                    x_cd = x_try.copy()
                            except Exception:
                                pass
                
                # Update best from coordinate descent
                if f_cd < self.best_f:
                    self.best_f = f_cd
                    self.best_x = x_cd.copy()
                
                # L-BFGS-B polish from the coordinate descent result
                if self.call_count < budget - 5 and time.time() - self.start_time < self.time_limit:
                    remaining_for_lbfgs = min(calls_per_opt, budget - self.call_count - 2)
                    if remaining_for_lbfgs > 10:
                        try:
                            res = minimize(wrapped_func, x_cd, method='L-BFGS-B', bounds=bounds,
                                           options={'maxfun': remaining_for_lbfgs, 'maxiter': 150})
                            if res.fun < self.best_f:
                                self.best_f = res.fun
                                self.best_x = np.clip(res.x, -5.0, 5.0).copy()
                        except Exception:
                            pass

        # Phase 3: Final Polish on the absolute best
        if self.call_count < budget - 10 and self.best_x is not None and time.time() - self.start_time < self.time_limit:
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
