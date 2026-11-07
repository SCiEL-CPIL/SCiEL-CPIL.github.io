"""Improved optimizer for BBOB f16 (Weierstrass) with coordinate-adaptive steps and local polish."""
import numpy as np
import time


class Optimizer:
    """Coordinate-adaptive (1+1)-ES with periodic gradient-based polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize
        x = np.zeros(dim)  # start at center
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # Per-dimension step sizes
        sigma = np.full(dim, 1.0)
        # Track success rate per dimension over recent moves
        success_count = np.zeros(dim, dtype=int)
        total_count = np.zeros(dim, dtype=int)
        window = 20  # recent moves to track

        # For gradient estimation
        eps_fd = 1e-4

        def time_left():
            return deadline - time.time()

        # Phase 1: Coordinate-adaptive exploration
        call_count = 1
        polish_interval = 50  # do a polish every 50 calls
        last_polish_call = 0

        for i in range(budget - 1):
            if time_left() < 5.0:
                break

            # Decide if we do a polish phase
            if i - last_polish_call >= polish_interval and time_left() > 30.0:
                # Gradient-based local polish
                x, fx = self._polish(func, x, fx, dim, eps_fd, rng, time_left, deadline)
                if fx < best_f:
                    best_x = x.copy()
                    best_f = fx
                call_count += 10  # approximate calls used in polish
                last_polish_call = i
                # Reset success tracking
                success_count[:] = 0
                total_count[:] = 0
                continue

            # Generate a perturbation: pick one random dimension to move
            d = rng.integers(0, dim)
            step = sigma[d]
            direction = rng.choice([-1, 1])

            y = x.copy()
            y[d] += direction * step
            y = np.clip(y, -5, 5)
            fy = func(y)
            call_count += 1

            if fy < fx:
                x = y
                fx = fy
                if fx < best_f:
                    best_x = x.copy()
                    best_f = fx
                # Update success rate for this dimension
                success_count[d] += 1
                total_count[d] += 1
                # Increase step size on success
                sigma[d] *= 1.2
            else:
                total_count[d] += 1
                # Decrease step size on failure
                sigma[d] *= 0.85

            # Periodically rebalance step sizes based on success rate
            if total_count[d] >= window:
                rate = success_count[d] / max(total_count[d], 1)
                if rate > 0.6:
                    sigma[d] *= 1.5
                elif rate < 0.3:
                    sigma[d] *= 0.7
                success_count[d] = 0
                total_count[d] = 0

            # Clamp sigma
            sigma = np.clip(sigma, 1e-6, 10.0)

        # Final polish with remaining budget
        remaining = budget - call_count
        if remaining > 5 and time_left() > 5.0:
            x, fx = self._polish(func, x, fx, dim, eps_fd, rng, time_left, deadline, max_calls=remaining)
            if fx < best_f:
                best_x = x.copy()
                best_f = fx

        return best_x, best_f

    def _polish(self, func, x, fx, dim, eps, rng, time_left, deadline, max_calls=10):
        """Gradient-based local polish using finite differences."""
        calls = 0

        for _ in range(5):  # up to 5 polish iterations
            if time_left() < 2.0 or calls >= max_calls:
                break

            # Estimate gradient
            grad = np.zeros(dim)
            for d in range(dim):
                if time_left() < 2.0 or calls >= max_calls:
                    break
                xp = x.copy()
                xp[d] += eps
                xp = np.clip(xp, -5, 5)
                fp = func(xp)
                calls += 1

                xm = x.copy()
                xm[d] -= eps
                xm = np.clip(xm, -5, 5)
                fm = func(xm)
                calls += 1

                grad[d] = (fp - fm) / (2 * eps)

            # Check if gradient is small
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-8:
                break

            # Adaptive step size
            step_size = min(0.5 / (gnorm + 1e-8), 1.0)

            # Try a few line search steps
            best_local = x.copy()
            best_local_f = fx
            for alpha in [step_size, step_size * 0.5, step_size * 0.25, step_size * 0.1]:
                if time_left() < 2.0 or calls >= max_calls:
                    break
                y = np.clip(x - alpha * grad, -5, 5)
                fy = func(y)
                calls += 1
                if fy < best_local_f:
                    best_local = y
                    best_local_f = fy

            if best_local_f < fx:
                x = best_local
                fx = best_local_f
                eps *= 0.5  # shrink finite difference step
            else:
                break

        return x, fx
