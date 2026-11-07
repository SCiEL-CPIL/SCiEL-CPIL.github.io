"""Optimized Weierstrass (f16) optimizer with coordinate-adaptive ES + gradient polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-lite with coordinate-wise step sizes and periodic gradient polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget

        # Initialize
        x_best = np.zeros(dim)
        fx_best = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            return f

        # Initial exploration: try a few random points
        for _ in range(min(10, budget - 1)):
            x_try = rng.uniform(-5, 5, dim)
            f_try = evaluate(x_try)
            if f_try < fx_best:
                fx_best = f_try
                x_best = x_try.copy()

        # Coordinate-wise step sizes
        sigma = np.ones(dim) * 2.0
        sigma_min = 1e-8
        sigma_max = 10.0

        # Success counters per dimension
        window = 20
        successes = [0] * dim
        total_attempts = [0] * dim

        remaining = budget - calls
        polish_interval = 100
        last_polish_call = calls

        for it in range(remaining):
            if time.time() > deadline:
                break

            # Periodic gradient polish
            if (calls - last_polish_call) >= polish_interval and (budget - calls) > 2 * dim:
                # Finite-difference gradient estimation
                h = 1e-4
                grad = np.zeros(dim)
                for d in range(dim):
                    x_plus = x_best.copy()
                    x_minus = x_best.copy()
                    x_plus[d] += h
                    x_minus[d] -= h
                    fp = evaluate(x_plus)
                    fm = evaluate(x_minus)
                    grad[d] = (fp - fm) / (2 * h)

                # Newton-like step with adaptive damping
                if np.linalg.norm(grad) > 1e-10:
                    # Estimate Hessian diagonal via second differences (cheap approximation)
                    # Use simple gradient descent with adaptive step
                    lr = 0.1
                    for _ in range(5):
                        if time.time() > deadline:
                            break
                        x_new = x_best - lr * grad
                        x_new = np.clip(x_new, -5, 5)
                        f_new = evaluate(x_new)
                        if f_new < fx_best:
                            fx_best = f_new
                            x_best = x_new.copy()
                            # Recompute gradient
                            for d in range(dim):
                                x_plus = x_best.copy()
                                x_minus = x_best.copy()
                                x_plus[d] += h
                                x_minus[d] -= h
                                fp = evaluate(x_plus)
                                fm = evaluate(x_minus)
                                grad[d] = (fp - fm) / (2 * h)
                            if np.linalg.norm(grad) < 1e-6:
                                break
                            lr *= 0.8
                        else:
                            lr *= 0.5
                            if lr < 1e-6:
                                break

                last_polish_call = calls

            # Coordinate-adaptive ES step
            # For each dimension, decide to perturb based on current sigma
            y = x_best.copy()
            step_taken = [False] * dim

            for d in range(dim):
                if sigma[d] < sigma_min:
                    continue
                # Random sign
                sign = 1.0 if rng.random() < 0.5 else -1.0
                delta_d = sign * sigma[d] * rng.standard_normal()
                y[d] = np.clip(x_best[d] + delta_d, -5.0, 5.0)

            fy = evaluate(y)

            if fy < fx_best:
                # Determine which dimensions contributed
                # Simple: accept and adjust sigmas
                for d in range(dim):
                    if abs(y[d] - x_best[d]) > sigma_min:
                        successes[d] += 1
                        total_attempts[d] += 1
                        # Adaptive step size based on success rate
                        if total_attempts[d] >= window:
                            rate = successes[d] / total_attempts[d]
                            if rate > 0.6:
                                sigma[d] = min(sigma[d] * 1.3, sigma_max)
                            elif rate < 0.4:
                                sigma[d] = max(sigma[d] * 0.7, sigma_min)
                            # Reset counters
                            successes[d] = 0
                            total_attempts[d] = 0

                x_best = y.copy()
                fx_best = fy
            else:
                for d in range(dim):
                    if abs(y[d] - x_best[d]) > sigma_min:
                        total_attempts[d] += 1
                        if total_attempts[d] >= window:
                            rate = successes[d] / total_attempts[d]
                            if rate < 0.4:
                                sigma[d] = max(sigma[d] * 0.7, sigma_min)
                            successes[d] = 0
                            total_attempts[d] = 0

            # Occasionally try larger jumps to escape local minima
            if it % 50 == 0 and (budget - calls) > 50:
                x_jump = x_best + rng.uniform(-2, 2, dim)
                x_jump = np.clip(x_jump, -5, 5)
                f_jump = evaluate(x_jump)
                if f_jump < fx_best:
                    fx_best = f_jump
                    x_best = x_jump.copy()

        # Final polish with very tight steps
        if (budget - calls) > dim and time.time() < deadline:
            h = 1e-5
            for _ in range(3):
                if time.time() > deadline or (budget - calls) < 2 * dim:
                    break
                grad = np.zeros(dim)
                for d in range(dim):
                    x_plus = x_best.copy()
                    x_minus = x_best.copy()
                    x_plus[d] += h
                    x_minus[d] -= h
                    fp = evaluate(x_plus)
                    fm = evaluate(x_minus)
                    grad[d] = (fp - fm) / (2 * h)

                if np.linalg.norm(grad) < 1e-8:
                    break

                # Take a small step
                lr = 0.01
                x_new = x_best - lr * grad
                x_new = np.clip(x_new, -5, 5)
                f_new = evaluate(x_new)
                if f_new < fx_best:
                    fx_best = f_new
                    x_best = x_new.copy()

        return x_best, fx_best
# EVOLVE-BLOCK-END
