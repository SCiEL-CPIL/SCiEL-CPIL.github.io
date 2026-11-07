"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin) using multi-start (1+1)-ES with momentum."""
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with momentum for multimodal functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def __call__(self, func):
        self.deadline = time.time() + 300.0  # Safety margin before 330s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Multi-start strategy: try several initial points to escape local minima
        # For Lunacek, the global minimum is often near the origin or shifted
        # We'll use a combination of center, random points, and a few structured starts

        starts = [np.zeros(n)]
        # Add some random starts
        for _ in range(5):
            starts.append(rng.uniform(-4, 4, size=n))
        # Add a few structured starts near potential minima
        starts.append(np.full(n, 2.0))
        starts.append(np.full(n, -2.0))
        starts.append(np.array([4.0] * n))
        starts.append(np.array([-4.0] * n))

        best_x = None
        best_f = np.inf
        best_trajectory = None  # (x, f, momentum, sigma) for best start

        for start_idx, x0 in enumerate(starts):
            if self.calls >= budget or time.time() > self.deadline:
                break

            # Check if we have enough budget for a reasonable run
            remaining = budget - self.calls
            if remaining < 20:
                break

            # Run (1+1)-ES with momentum from this start
            x = x0.copy()
            fx = func(x)
            self.calls += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Initialize momentum and step size
            momentum = np.zeros(n)
            sigma = 1.0
            success_count = 0
            total_steps = 0

            # (1+1)-ES with momentum
            while self.calls < budget and time.time() < self.deadline:
                # Generate candidate
                z = rng.standard_normal(n)
                step = sigma * z
                # Apply momentum
                step = 0.5 * momentum + 0.5 * step
                momentum = 0.5 * momentum + 0.5 * step

                x_new = x + step
                x_new = np.clip(x_new, -5, 5)

                fx_new = func(x_new)
                self.calls += 1
                total_steps += 1

                if fx_new < fx:
                    # Accept
                    x = x_new
                    fx = fx_new
                    success_count += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                        best_trajectory = (x.copy(), fx, momentum.copy(), sigma)
                    # Increase sigma slightly
                    sigma *= 1.05
                else:
                    # Reject
                    # Decrease sigma
                    sigma *= 0.8
                    # Decay momentum
                    momentum *= 0.5

                # Check for stagnation: if no success in last 20 steps, restart from best
                if total_steps > 20 and success_count == 0:
                    # Try a different strategy: jump to a random point
                    x_jump = best_x + rng.normal(0, 2.0, size=n)
                    x_jump = np.clip(x_jump, -5, 5)
                    fx_jump = func(x_jump)
                    self.calls += 1
                    if fx_jump < fx:
                        x = x_jump
                        fx = fx_jump
                        success_count = 1
                    else:
                        # Reset momentum
                        momentum = np.zeros(n)
                        sigma = 1.0
                        success_count = 0
                    break

                # Check if we've converged
                if total_steps > 50 and np.linalg.norm(step) < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
