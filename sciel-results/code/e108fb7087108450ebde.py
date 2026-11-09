import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 285.0

        budget = self.budget
        d = self.dim
        calls = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline or calls >= budget:
                raise StopIteration()
            f = float(func(np.asarray(x, dtype=np.float64)))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = np.asarray(x, dtype=np.float64).copy()
            return f

        x = np.zeros(d)
        try:
            f_x = safe_func(x)
        except StopIteration:
            return best_x, best_f

        # Phase 1: Global (1+1)-ES
        # Parameters: growth 1.2, decay 0.9, stall_limit 50
        # Explore budget: 60%
        sigma = 1.0
        stall_count = 0
        stall_limit = 50
        growth = 1.2
        decay = 0.9
        es_limit = int(budget * 0.6)

        while calls < es_limit and time.time() < deadline:
            step = rng.normal(0, 1, d)
            x_candidate = x + sigma * step
            
            try:
                f_candidate = safe_func(x_candidate)
            except StopIteration:
                break

            if f_candidate < f_x:
                x = x_candidate
                f_x = f_candidate
                sigma *= growth
                stall_count = 0
            else:
                sigma *= decay
                stall_count += 1
                
            if stall_count >= stall_limit:
                sigma = 2.0
                stall_count = 0

        # Phase 2: Local Refinement (Local (1+1)-ES)
        if best_x is not None and calls < budget:
            x = best_x.copy()
            f_x = best_f
            sigma = 0.1
            
            while calls < budget and time.time() < deadline:
                step = rng.normal(0, 1, d)
                x_candidate = x + sigma * step
                
                try:
                    f_candidate = safe_func(x_candidate)
                except StopIteration:
                    break

                if f_candidate < f_x:
                    x = x_candidate
                    f_x = f_candidate
                    sigma *= 1.1
                else:
                    sigma *= 0.95
                    if sigma < 1e-6:
                        break

        return best_x, best_f
