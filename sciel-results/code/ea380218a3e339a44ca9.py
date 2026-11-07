"""Rastrigin optimizer: Custom CMA-ES with adaptive restarts + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Restarted Cyclic Coordinate Descent with Golden Section Search for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Golden Section Search for 1D minimization
        def golden_section_search(func, x_current, dim_idx, a, b, max_iter=15):
            nonlocal calls_made
            if calls_made >= budget:
                return x_current
            
            phi = (np.sqrt(5) - 1) / 2  # ~0.618
            c = b - phi * (b - a)
            d = a + phi * (b - a)
            
            x_c = x_current.copy()
            x_d = x_current.copy()
            x_c[dim_idx] = c
            x_d[dim_idx] = d
            
            f_c = counted_func(x_c)
            f_d = counted_func(x_d)
            
            if calls_made >= budget:
                return x_current

            for _ in range(max_iter):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                if f_c < f_d:
                    b = d
                    d = c
                    f_d = f_c
                    c = b - phi * (b - a)
                    x_c = x_current.copy()
                    x_c[dim_idx] = c
                    f_c = counted_func(x_c)
                else:
                    a = c
                    c = d
                    f_c = f_d
                    d = a + phi * (b - a)
                    x_d = x_current.copy()
                    x_d[dim_idx] = d
                    f_d = counted_func(x_d)
                
                if abs(b - a) < 1e-6:
                    break

            if f_c < f_d:
                x_current[dim_idx] = c
            else:
                x_current[dim_idx] = d
            return x_current

        # Phase 1: Initial Sampling
        n_init = 10
        for i in range(n_init):
            if calls_made >= budget or time.time() > deadline:
                break
            if i == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-5, 5, dim)
            counted_func(x)

        # Phase 2: Restarted Cyclic Coordinate Descent
        current_x = best_x.copy()
        current_f = best_f
        
        restarts = 0
        max_restarts = 5
        
        while calls_made < budget - 10 and time.time() < deadline - 5:
            improved = False
            f_before_cycle = current_f
            
            for d_idx in range(dim):
                if calls_made >= budget or time.time() > deadline:
                    break
                
                a = current_x[d_idx] - 1.0
                b = current_x[d_idx] + 1.0
                a = max(a, -5.0)
                b = min(b, 5.0)
                
                if a >= b:
                    continue
                    
                current_x = golden_section_search(counted_func, current_x, d_idx, a, b, max_iter=15)
                
            f_after_cycle = counted_func(current_x)
            
            if f_after_cycle < best_f:
                best_f = f_after_cycle
                best_x = current_x.copy()
                current_f = best_f
            else:
                current_f = f_after_cycle
            
            if current_f < f_before_cycle - 1e-4:
                improved = True
            else:
                improved = False
            
            if not improved:
                restarts += 1
                if restarts > max_restarts:
                    break
                
                new_x = rng.uniform(-3, 3, dim)
                f_new = counted_func(new_x)
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = new_x.copy()
                    current_x = new_x.copy()
                    current_f = f_new
                else:
                    current_x = new_x.copy()
                    current_f = f_new

        return best_x, best_f
# EVOLVE-BLOCK-END
