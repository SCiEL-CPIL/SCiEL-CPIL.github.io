"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with fallback to Nelder-Mead, with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        # Use CMA-ES via scipy's minimize with 'Nelder-Mead' as fallback
        # Actually, let's implement a proper CMA-ES or use a better approach
        
        # Strategy: Use multiple random starts with local optimization
        # Given the BBOB f4 (Büche-Rastrigin) which is multimodal, we need
        # global exploration + local exploitation
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def tracked_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError()
            calls += 1
            if calls > self.budget:
                raise TimeoutError()
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Phase 1: CMA-ES-like search with adaptive step size
        # Initialize with a spread-out point
        x = rng.uniform(-5, 5, self.dim)
        fx = tracked_func(x)
        best_x = x.copy()
        best_f = fx
        
        sigma = 1.5
        success_count = 0
        total_count = 0
        target_success = 0.2  # 1/5 rule
        
        while calls < self.budget and time.time() < deadline:
            # Adapt sigma based on success rate
            if total_count > 5:
                success_rate = success_count / total_count
                if success_rate > target_success * 1.5:
                    sigma *= 1.1
                elif success_rate < target_success * 0.5:
                    sigma *= 0.9
                # Reset counters periodically
                if total_count > 20:
                    success_count = 0
                    total_count = 0
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = tracked_func(y)
            total_count += 1
            
            if fy < fx:
                x, fx = y, fy
                success_count += 1
            else:
                # Sometimes take a larger exploratory step
                if rng.random() < 0.05:
                    y2 = np.clip(x + sigma * 3 * rng.standard_normal(self.dim), -5, 5)
                    fy2 = tracked_func(y2)
                    if fy2 < fx:
                        x, fx = y2, fy2
                        sigma *= 0.9
        
        # Phase 2: Local refinement using Nelder-Mead if we have time
        if best_x is not None and time.time() < deadline - 10:
            try:
                result = minimize(
                    tracked_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': min(self.budget - calls, 200),
                             'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
