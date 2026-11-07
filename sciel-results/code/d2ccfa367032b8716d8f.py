"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase hybrid: dense radial/axis sweep to find the needle, then aggressive (1+1)-ES."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        # Track best
        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Dense deterministic sweep to find the narrow axis of Discus
        # Discus f11: f(x) = 0.5 * (||x||^2 + 2 * max(0, |x_i|)^2) type structure
        # Actually BBOB f11 (Discus): f(x) = 0.5 * (||x||^2 + 2 * max_i |x_i|^2)
        # The needle is along the axis where the max component is large.
        # Strategy: sample along each coordinate axis at multiple radii, plus radial directions.

        # First, evaluate origin
        evaluate(np.zeros(dim))

        # Axis-based sweep: for each axis, try positive and negative directions at several radii
        radii = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        for axis in range(dim):
            for r in radii:
                if calls >= budget or time.time() > deadline:
                    break
                # Positive direction
                x = np.zeros(dim)
                x[axis] = r
                evaluate(x)
                # Negative direction
                x[axis] = -r
                if calls < budget and time.time() < deadline:
                    evaluate(x)
            if calls >= budget or time.time() > deadline:
                break

        # Radial sweep in random directions
        n_radial = 20
        for i in range(n_radial):
            if calls >= budget or time.time() > deadline:
                break
            direction = rng.standard_normal(dim)
            direction /= np.linalg.norm(direction)
            for r in [1.0, 2.0, 3.0, 4.0, 5.0]:
                if calls >= budget or time.time() > deadline:
                    break
                evaluate(direction * r)

        # Phase 2: Aggressive (1+1)-ES with adaptive step size
        # Start from the best point found
        x = best_x.copy() if best_x is not None else np.zeros(dim)
        fx = best_f

        # Initialize sigma based on the spread of evaluations
        # Use a moderately large initial sigma for exploration
        sigma = 1.0
        success_window = []
        window_size = 10

        remaining = budget - calls
        for _ in range(remaining):
            if time.time() > deadline:
                break

            # Generate perturbation
            perturbation = rng.standard_normal(dim)
            y = np.clip(x + sigma * perturbation, -5.0, 5.0)
            fy = evaluate(y)

            # Adapt sigma using 1/5 success rule (simplified)
            success_window.append(1 if fy < fx else 0)
            if len(success_window) > window_size:
                success_window.pop(0)
            success_rate = np.mean(success_window)
            if success_rate > 0.5:
                sigma *= 1.2
            elif success_rate < 0.2:
                sigma *= 0.8
            else:
                sigma *= 0.95

            # Clamp sigma
            sigma = max(sigma, 0.01)
            sigma = min(sigma, 10.0)

            if fy < fx:
                x, fx = y, fy

        return best_x, best_f
# EVOLVE-BLOCK-END
