import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s margin before 300s limit
        
        rng = np.random.default_rng(123)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        
        def evaluate(x):
            nonlocal best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Phase 1: Global search with large steps (first ~100 calls)
        # Use a random walk with occasional large jumps
        x = rng.uniform(-5, 5, dim)
        evaluate(x)
        calls = 1
        
        # Large step exploration
        for i in range(99):
            if time.time() > deadline:
                break
            # Mix of large and medium steps
            step_size = 3.0 * (0.5 ** (i // 25))  # decay from 3.0 to 0.375
            y = x + step_size * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = evaluate(y)
            calls += 1
            if fy < best_f:
                x = y
        
        # Phase 2: Local refinement using coordinate descent
        # For each dimension, do a golden-section search
        phi = (1 + np.sqrt(5)) / 2
        inv_phi = 1 / phi
        inv_phi2 = inv_phi * inv_phi
        
        def golden_search_1d(func_eval, x_fixed, dim_idx, lo, hi, max_iter=20):
            """Golden section search on one dimension."""
            a = lo
            b = hi
            c = b - inv_phi * (b - a)
            d = a + inv_phi * (b - a)
            
            x_c = x_fixed.copy()
            x_c[dim_idx] = c
            x_d = x_fixed.copy()
            x_d[dim_idx] = d
            
            fc = func_eval(x_c)
            fd = func_eval(x_d)
            
            for _ in range(max_iter):
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - inv_phi * (b - a)
                    x_c = x_fixed.copy()
                    x_c[dim_idx] = c
                    fc = func_eval(x_c)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + inv_phi * (b - a)
                    x_d = x_fixed.copy()
                    x_d[dim_idx] = d
                    fd = func_eval(x_d)
            
            opt = (a + b) / 2
            return opt
        
        # Phase 2a: Coordinate descent with golden section
        x_cur = best_x.copy()
        fc = best_f
        
        for iteration in range(10):
            if time.time() > deadline:
                break
            improved = False
            for d in range(dim):
                if time.time() > deadline:
                    break
                # Determine search range around current value
                center = x_cur[d]
                # Use a range that covers the box but centered on current
                lo = max(-5, center - 4.0)
                hi = min(5, center + 4.0)
                
                # If at boundary, shrink range
                if center <= -4.9:
                    lo, hi = -5, -1
                elif center >= 4.9:
                    lo, hi = 1, 5
                
                opt_val = golden_search_1d(func_eval, x_cur, d, lo, hi, max_iter=15)
                x_new = x_cur.copy()
                x_new[d] = opt_val
                f_new = func_eval(x_new)
                
                if f_new < fc - 1e-12:
                    x_cur = x_new
                    fc = f_new
                    improved = True
            
            if not improved:
                break
        
        # Phase 3: Fine-tuning with small perturbations
        if time.time() < deadline:
            sigma = 0.1
            for _ in range(50):
                if time.time() > deadline:
                    break
                y = x_cur + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func_eval(y)
                if fy < best_f:
                    x_cur = y
                    best_f = fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8
        
        return best_x, best_f
