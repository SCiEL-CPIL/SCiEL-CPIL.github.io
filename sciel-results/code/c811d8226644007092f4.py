"""Rewritten optimizer for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(mu+lambda)-ES with anisotropic covariance and stagnation-based restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s kill

        # Global best
        best_x = np.zeros(self.dim)
        best_f = np.inf
        total_calls = 0
        budget = self.budget
        dim = self.dim

        def call_func(x):
            nonlocal total_calls
            total_calls += 1
            return func(x)

        # Initial broad exploration with a small population to find a good starting point
        # Use a few random points to get an initial landscape understanding
        rng = np.random.default_rng(42)

        # Phase 1: Initial exploration (use ~10% of budget)
        n_init = min(50, budget // 5)
        candidates = []
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = call_func(x)
            candidates.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort and take the best as starting point for ES
        candidates.sort(key=lambda t: t[0])
        x_mean = candidates[0][1].copy()
        f_mean = candidates[0][0]

        # Estimate scale from initial spread
        inits = np.array([c[1] for c in candidates[:min(10, len(candidates))]])
        f_inits = np.array([c[0] for c in candidates[:min(10, len(candidates))]])
        if len(inits) > 2 and np.ptp(f_inits) > 1e-10:
            # Rough scale estimate
            scale_est = np.std(inits, axis=0)
            scale_est = np.where(scale_est < 0.1, 0.5, scale_est)
        else:
            scale_est = np.ones(dim) * 1.0

        # Phase 2: Main ES loop
        # Use (mu+lambda)-ES with mu=10, lambda=50
        mu = 10
        lam = 50

        # Anisotropic covariance: start with diagonal based on initial scale
        # For step ellipsoidal, the condition number can be high
        # Initialize sigma per dimension
        sigma = np.ones(dim) * 0.5

        # Track progress for stagnation detection
        stall_count = 0
        stall_threshold = 80  # restart if no improvement for 80 generations
        gen = 0
        best_f_history = best_f

        while total_calls < budget and time.time() < deadline:
            # Generate population
            offspring = []
            for i in range(lam):
                # Anisotropic step: different step sizes per dimension
                noise = rng.standard_normal(dim)
                x_child = x_mean + sigma * noise
                x_child = np.clip(x_child, -5, 5)
                f_child = call_func(x_child)
                offspring.append((f_child, x_child))

                if total_calls >= budget:
                    break

            if not offspring:
                break

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            if offspring[0][0] < best_f:
                best_f = offspring[0][0]
                best_x = offspring[0][1].copy()

            # Stagnation check
            if offspring[0][0] < best_f_history - 1e-8:
                stall_count = 0
                best_f_history = offspring[0][0]
            else:
                stall_count += 1

            # Select top mu to compute new mean
            selected = [o[1] for o in offspring[:mu]]
            x_mean = np.mean(selected, axis=0)

            # Adapt sigma per dimension based on success rate
            # Simple adaptation: if best improved, increase sigma slightly; else decrease
            improved = offspring[0][0] < f_mean - 1e-8
            if improved:
                sigma *= 1.1
            else:
                sigma *= 0.9

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.01, 5.0)

            f_mean = offspring[0][0]
            gen += 1

            # Restart if stagnated
            if stall_count >= stall_threshold:
                # Local restart: perturb the best point
                perturbation = rng.standard_normal(dim) * 0.5
                x_mean = np.clip(best_x + perturbation, -5, 5)
                f_mean = best_f
                sigma = np.ones(dim) * 1.0
                stall_count = 0

                # If still stuck, do a global restart
                if gen > 100:
                    x_mean = rng.uniform(-4, 4, dim)
                    f_mean = call_func(x_mean)
                    sigma = np.ones(dim) * 1.5
                    if f_mean < best_f:
                        best_f = f_mean
                        best_x = x_mean.copy()

            # Check time
            if time.time() > deadline - 5.0:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
