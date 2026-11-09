"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        deadline = time.time() + 320.0  # safety margin before 330s kill

        # Start at the center of the box
        x = np.zeros(self.dim)
        fx = func(x)
        calls = 1

        # Initial step size
        sigma = 2.0

        # Adaptation factors for 1/5 success rule (gentler for stability)
        c_inc = 1.1
        c_dec = 1.0 / 1.1

        # Track success rate over a window for better adaptation
        success_window = []
        window_size = 20

        for _ in range(self.budget - 1):
            # Deadline guard
            if time.time() > deadline:
                break

            # Generate trial point
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                success_window.append(1)
            else:
                success_window.append(0)

            # Adaptive sigma based on recent success rate
            if len(success_window) >= window_size:
                success_rate = np.mean(success_window[-window_size:])
                if success_rate > 0.3:
                    sigma *= 1.05
                elif success_rate < 0.2:
                    sigma *= 1.0 / 1.05
                # else keep sigma same

            # Prevent sigma from becoming too small or too large
            sigma = np.clip(sigma, 1e-8, 10.0)

        # Final polish: Nelder-Mead if we have time
        if time.time() < deadline - 5.0:
            try:
                from scipy.optimize import minimize
                res = minimize(lambda v: func(v), x, method='Nelder-Mead',
                               options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < fx:
                    x, fx = res.x, res.fun
            except Exception:
                pass

        return x, fx
# EVOLVE-BLOCK-END
