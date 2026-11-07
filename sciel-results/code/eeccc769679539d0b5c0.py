import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None
        self.call_count = 0

    def __call__(self, func):
        self.start_time = time.time()
        self.call_count = 0
        best_x = None
        best_f = float('inf')
        
        deadline = self.start_time + 325  # 5s safety margin before 330s kill
        
        def tracked_func(x):
            """Function wrapper that tracks calls and enforces limits."""
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if self.call_count >= self.budget:
                raise ValueError("Budget exceeded")
            self.call_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            return func(x)
        
        # Phase 1: Stratified random sampling for initial diversity
        n_initial = min(25, self.budget // 8)
        rng = np.random.RandomState(42)
        
        # Create stratified points
        points = np.zeros((n_initial, self.dim))
        for d in range(self.dim):
            strata = np.linspace(-5.0, 5.0, n_initial + 1)
            for i in range(n_initial):
                points[i, d] = rng.uniform(strata[i], strata[i+1])
        
        for x in points:
            if time.time() > deadline or self.call_count >= self.budget:
                break
            try:
                fx = tracked_func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            except Exception:
                break
        
        if best_x is None:
            try:
                fx = tracked_func(np.zeros(self.dim))
                best_f = fx
                best_x = np.zeros(self.dim)
            except Exception:
                pass
            return best_x, best_f
        
        # Phase 2: Nelder-Mead from best point with tight tolerance
        if self.budget - self.call_count > 30 and time.time() < deadline:
            try:
                def nm_wrapper(x):
                    if time.time() > deadline:
                        raise TimeoutError("Time limit")
                    if self.call_count >= self.budget:
                        raise ValueError("Budget exceeded")
                    self.call_count += 1
                    x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                    return func(x)
                
                res = minimize(
                    nm_wrapper,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12,
                             'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # Phase 3: Additional restarts from diverse random points
        remaining = self.budget - self.call_count
        if remaining > 20 and time.time() < deadline:
            n_restarts = min(5, remaining // 15)
            rng2 = np.random.RandomState(123)
            
            for _ in range(n_restarts):
                if self.call_count >= self.budget - 10 or time.time() > deadline:
                    break
                
                # Random restart in [-4, 4]^5
                x0 = rng2.uniform(-4.0, 4.0, size=self.dim)
                try:
                    fx0 = tracked_func(x0)
                    if fx0 < best_f:
                        best_f = fx0
                        best_x = x0.copy()
                except Exception:
                    break
                
                if self.budget - self.call_count > 10 and time.time() < deadline:
                    try:
                        def nm_wrapper2(x):
                            if time.time() > deadline:
                                raise TimeoutError("Time limit")
                            if self.call_count >= self.budget:
                                raise ValueError("Budget exceeded")
                            self.call_count += 1
                            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                            return func(x)
                        
                        res = minimize(
                            nm_wrapper2,
                            x0,
                            method='Nelder-Mead',
                            options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10,
                                     'adaptive': True}
                        )
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x.copy()
                    except Exception:
                        pass
        
        return best_x, best_f
