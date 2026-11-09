import numpy as np
import scipy.optimize as so
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = 0
        self.time_limit = 275  # Safety margin for 300s limit

    def __call__(self, func):
        self.start_time = time.time()
        rng = np.random.default_rng(42)
        
        def check_deadline():
            return time.time() - self.start_time > self.time_limit
        
        def evaluate(x):
            if self.calls >= self.budget or check_deadline():
                return None
            self.calls += 1
            return func(x)
        
        best_x = None
        best_f = np.inf
        
        # Phase 1: Latin Hypercube Sampling for initial exploration
        n_init = min(40, self.budget // 4)
        lhs = np.zeros((n_init, self.dim))
        for i in range(self.dim):
            perm = rng.permutation(n_init)
            lhs[:, i] = (perm + 0.5) / n_init
        
        x_init = lhs * 10 - 5
        
        for x in x_init:
            fx = evaluate(x)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            return None, np.inf
        
        # Phase 2: Local refinement using L-BFGS-B from top candidates
        # Collect top candidates from initial sampling
        candidates = []
        for x in x_init:
            fx = evaluate(x)
            if fx is None:
                break
            candidates.append((x, fx))
        
        # Sort by fitness and pick top candidates for local search
        candidates.sort(key=lambda t: t[1])
        
        # Try local search from top 5 candidates (or fewer if budget is tight)
        n_local = min(5, self.budget - self.calls, 10)
        
        for i in range(n_local):
            if self.calls >= self.budget or check_deadline():
                break
            
            start_x = candidates[i][0].copy()
            
            # Wrap func for L-BFGS-B with call counting
            call_count = [0]
            max_calls = min(80, self.budget - self.calls)
            
            def local_func(x, count=[0]):
                if count[0] >= max_calls or check_deadline():
                    return best_f  # Return best known to encourage stopping
                count[0] += 1
                self.calls += 1
                return func(x)
            
            try:
                res = so.minimize(
                    local_func, start_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # Phase 3: Adaptive perturbation and further local search
        # Use remaining budget to explore around best_x
        while self.calls < self.budget and not check_deadline():
            # Generate perturbation around best_x
            scale = max(0.1, (best_f - 0) * 0.01) if best_f > 0 else 0.5
            x_new = best_x + rng.normal(0, scale, self.dim)
            x_new = np.clip(x_new, -5, 5)
            
            fx = evaluate(x_new)
            if fx is None:
                break
            
            if fx < best_f:
                best_f = fx
                best_x = x_new.copy()
            
            # If we found improvement, do a quick local polish
            if fx < best_f + 1e-6 and self.calls < self.budget - 20:
                call_count = [0]
                max_calls = min(30, self.budget - self.calls)
                
                def polish_func(x):
                    if call_count[0] >= max_calls or check_deadline():
                        return best_f
                    call_count[0] += 1
                    self.calls += 1
                    return func(x)
                
                try:
                    res = so.minimize(
                        polish_func, x_new,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={'maxiter': 20, 'ftol': 1e-10}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass
        
        # Phase 4: Final high-precision polish if budget remains
        if self.calls < self.budget - 10 and not check_deadline():
            call_count = [0]
            max_calls = min(50, self.budget - self.calls)
            
            def final_func(x):
                if call_count[0] >= max_calls or check_deadline():
                    return best_f
                call_count[0] += 1
                self.calls += 1
                return func(x)
            
            try:
                res = so.minimize(
                    final_func, best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        return best_x, best_f
