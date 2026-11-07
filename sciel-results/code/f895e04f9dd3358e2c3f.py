"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear function optimizer using gradient sign estimation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Track calls and best
        call_count = 0
        best_x = None
        best_f = np.inf
        
        def safe_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return best_f
            call_count += 1
            fx = func(np.clip(x, -5, 5))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx
        
        # Phase 1: Estimate gradient signs from origin
        origin = np.zeros(dim)
        f0 = safe_func(origin)
        
        delta = 1.0
        c_signs = np.zeros(dim)
        
        for i in range(dim):
            if call_count >= budget or time.time() > deadline:
                break
            x_pos = origin.copy()
            x_pos[i] += delta
            x_neg = origin.copy()
            x_neg[i] -= delta
            f_pos = safe_func(x_pos)
            f_neg = safe_func(x_neg)
            # c_i proportional to (f_pos - f_neg)
            c_signs[i] = f_pos - f_neg
        
        # Phase 2: Jump to estimated optimal vertex
        if call_count < budget and time.time() < deadline:
            # For minimization: if c_i > 0, go to -5; if c_i < 0, go to +5
            x_vertex = np.zeros(dim)
            for i in range(dim):
                if c_signs[i] > 0:
                    x_vertex[i] = -5.0
                elif c_signs[i] < 0:
                    x_vertex[i] = 5.0
                else:
                    x_vertex[i] = 0.0  # Indeterminate, stay at 0
            
            f_vertex = safe_func(x_vertex)
        
        # Phase 3: Local refinement around best point using (1+1)-ES
        if best_x is None:
            best_x = origin.copy()
            best_f = f0
        
        x = best_x.copy()
        fx = best_f
        sigma = 0.5
        
        # Adaptive (1+1)-ES for remaining budget
        while call_count < budget and time.time() < deadline:
            y = x + sigma * rng.standard_normal(dim)
            fy = safe_func(y)
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.85
            
            # Early stop if sigma becomes very small
            if sigma < 1e-6:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
