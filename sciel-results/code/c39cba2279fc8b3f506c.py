"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based adaptive ES with multi-start and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        # Safe wrapper to handle potential edge cases
        def safe_func(x):
            x = np.asarray(x, dtype=np.float64).ravel()
            if x.shape[0] != self.dim:
                x = np.resize(x, self.dim)
            x = np.clip(x, -5.0, 5.0)
            try:
                val = float(func(x))
            except Exception:
                val = 1e10
            if not np.isfinite(val):
                val = 1e10
            return val

        # Phase 1: Multi-start with diverse initializations
        # Use a mix of random points and structured grid-like points
        n_starts = 15
        starts = []
        # First start: origin
        starts.append(np.zeros(self.dim))
        # Structured points
        for i in range(1, n_starts):
            p = rng.uniform(-4.5, 4.5, self.dim)
            starts.append(p)

        best_x = None
        best_f = np.inf
        calls_used = 0

        for s in starts:
            if calls_used >= self.budget:
                break
            if time.time() > deadline:
                break
            f = safe_func(s)
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = s.copy()

        # Phase 2: Adaptive (mu+lambda)-ES with step-size adaptation
        # Population-based search around the best point
        mu = 4
        lam = 12
        sigma = 1.5
        x_best = best_x.copy()
        f_best = best_f

        # Track success rate for step-size adaptation (1/5 rule)
        success_count = 0
        total_evals_phase2 = 0

        while calls_used < self.budget:
            if time.time() > deadline:
                break

            # Generate offspring
            offspring = []
            for i in range(lam):
                if calls_used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                # CMA-like: use current best as center, add noise
                step = sigma * rng.standard_normal(self.dim)
                child = np.clip(x_best + step, -5.0, 5.0)
                f_child = safe_func(child)
                calls_used += 1
                total_evals_phase2 += 1
                offspring.append((f_child, child))

            if not offspring:
                break

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[0])
            # Take top mu
            for f_off, x_off in offspring[:mu]:
                if f_off < f_best:
                    f_best = f_off
                    x_best = x_off.copy()
                    success_count += 1

            # Step-size adaptation (1/5 rule)
            if total_evals_phase2 > 0:
                success_rate = success_count / total_evals_phase2
                if success_rate > 0.5:
                    sigma *= 1.1
                elif success_rate < 0.1:
                    sigma *= 0.8
                else:
                    sigma *= 0.95

            # Reset counters for next generation
            success_count = 0
            total_evals_phase2 = 0

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.01, 5.0)

        # Phase 3: Local refinement with very small steps
        if calls_used < self.budget and time.time() < deadline:
            sigma_local = 0.05
            for _ in range(min(50, self.budget - calls_used)):
                if time.time() > deadline:
                    break
                if calls_used >= self.budget:
                    break
                step = sigma_local * rng.standard_normal(self.dim)
                child = np.clip(x_best + step, -5.0, 5.0)
                f_child = safe_func(child)
                calls_used += 1
                if f_child < f_best:
                    f_best = f_child
                    x_best = x_off.copy()
                    sigma_local *= 0.9
                else:
                    sigma_local *= 1.1
                sigma_local = np.clip(sigma_local, 0.001, 0.5)

        return x_best, f_best
# EVOLVE-BLOCK-END
