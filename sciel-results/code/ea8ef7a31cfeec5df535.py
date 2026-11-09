"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Gradient-guided (1+1) optimizer with adaptive step sizing for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Adaptive step size
        sigma = 2.0
        min_sigma = 1e-8
        max_sigma = 10.0

        # Gradient estimation via finite differences
        def estimate_gradient(x, fx, step):
            grad = np.zeros(dim)
            for i in range(dim):
                x_plus = x.copy()
                x_plus[i] = min(5.0, x_plus[i] + step)
                x_minus = x.copy()
                x_minus[i] = max(-5.0, x_minus[i] - step)
                fp = func(x_plus)
                fm = func(x_minus)
                calls[0] += 2
                grad[i] = (fp - fm) / (x_plus[i] - x_minus[i])
            return grad

        # Use a mutable list to track calls
        calls = [1]

        # Main optimization loop
        for iteration in range(budget - 1):
            # Check time
            if time.time() > deadline:
                break

            # Estimate gradient
            grad = estimate_gradient(x, fx, sigma * 0.1)
            grad_norm = np.linalg.norm(grad)

            if grad_norm > 1e-10:
                # Gradient direction
                grad_dir = -grad / grad_norm
            else:
                # Random direction if gradient is flat
                grad_dir = rng.standard_normal(dim)
                grad_dir /= np.linalg.norm(grad_dir)

            # Adaptive step size based on gradient magnitude
            # In flat regions (small gradient), take larger steps
            # Near optimum (large gradient relative to distance), take smaller steps
            adaptive_factor = 1.0 / (1.0 + grad_norm * 0.1)
            step_size = sigma * adaptive_factor

            # Propose new point
            y = x + step_size * grad_dir
            y = np.clip(y, -5, 5)

            # Evaluate
            fy = func(y)
            calls[0] += 1

            if fy < fx:
                # Accept: move in gradient direction
                x = y
                fx = fy
                if fx < best_fx:
                    best_x = x.copy()
                    best_fx = fx
                # Increase step size on success
                sigma = min(max_sigma, sigma * 1.3)
            else:
                # Reject: reduce step size
                sigma = max(min_sigma, sigma * 0.7)

            # Occasional random exploration to escape local flat regions
            if rng.random() < 0.05:
                y = x + sigma * 2.0 * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls[0] += 1
                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_fx:
                        best_x = x.copy()
                        best_fx = fx
                    sigma = min(max_sigma, sigma * 1.2)

            # Check budget
            if calls[0] >= budget:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
