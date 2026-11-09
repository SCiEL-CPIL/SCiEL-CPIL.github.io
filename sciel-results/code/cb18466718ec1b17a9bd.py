"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, deadline guard, and multi-stage Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 3.0  # Larger initial sigma for better early exploration
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        # Track best solution for potential polish
        best_x = x.copy()
        best_fx = fx
        budget_used = 1

        # (1+1)-ES with 1/5 success rule
        for _ in range(self.budget - 1):
            if time.time() >= deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            budget_used += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87
            if fy < best_fx:
                best_x, best_fx = y.copy(), fy

        # Multi-stage Nelder-Mead polish with decreasing step sizes
        remaining = self.budget - budget_used
        if remaining > 10 and time.time() < deadline - 5:
            try:
                from scipy.optimize import minimize
                polish_calls = [0]
                max_polish_calls = min(remaining, 200)

                def polish_func(z):
                    if polish_calls[0] >= max_polish_calls:
                        return best_fx
                    polish_calls[0] += 1
                    z_clipped = np.clip(z, -5, 5)
                    return func(z_clipped)

                # Stage 1: Medium-step Nelder-Mead
                initial_step = 0.5
                result = minimize(
                    polish_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_polish_calls // 2,
                        'initial_simplex': np.array([
                            best_x + initial_step * e
                            for e in np.eye(self.dim)
                        ]),
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if result.fun < best_fx:
                    best_x = np.clip(result.x, -5, 5)
                    best_fx = result.fun

                # Stage 2: Fine-step Nelder-Mead for high-precision polish
                remaining2 = max_polish_calls - polish_calls[0]
                if remaining2 > 5 and time.time() < deadline - 3:
                    initial_step2 = 0.05
                    result2 = minimize(
                        polish_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': remaining2,
                            'initial_simplex': np.array([
                                best_x + initial_step2 * e
                                for e in np.eye(self.dim)
                            ]),
                            'xatol': 1e-10,
                            'fatol': 1e-12,
                            'adaptive': True
                        }
                    )
                    if result2.fun < best_fx:
                        best_x = np.clip(result2.x, -5, 5)
                        best_fx = result2.fun
            except Exception:
                pass  # If polish fails, keep the ES result

        return best_x, best_fx
# EVOLVE-BLOCK-END
