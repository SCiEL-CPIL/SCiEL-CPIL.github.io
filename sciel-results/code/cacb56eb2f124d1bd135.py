"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive sigma and time guard for Rastrigin-rotated f15."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _run_phase(self, func, rng, x0, fx0, calls_made, best_f, best_x, deadline,
                   budget, dim, sigma_start, sigma_inc, sigma_dec, max_stall,
                   max_steps, polish=False):
        """Run a (1+1)-ES from x0 for up to max_steps evaluations.
        Returns (new_x, new_fx, new_calls, best_f, best_x, terminated_stuck)."""
        x = x0.copy()
        fx = fx0
        sigma = sigma_start
        no_improve = 0
        steps = 0
        stuck = False

        for _ in range(max_steps):
            if calls_made >= budget:
                break
            if time.time() >= deadline:
                break

            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = func(y)
            calls_made += 1
            steps += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= sigma_inc
                no_improve = 0
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
            else:
                sigma *= sigma_dec
                no_improve += 1

            if polish:
                # Tighter stopping for polish
                if no_improve >= max(8, max_stall // 3) or sigma < 1e-6:
                    break
            else:
                if no_improve >= max_stall or sigma < 1e-4:
                    break

        return x, fx, calls_made, best_f, best_x, stuck

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin

        rng = np.random.default_rng(42)

        # Track global best
        best_f = np.inf
        best_x = np.zeros(dim)
        calls_made = 0

        # Phase 1: Broad exploration with multiple restarts
        # Use varied starting points
        n_restarts = 6
        sigma_start = 2.5
        sigma_inc = 1.3
        sigma_dec = 0.8
        max_stall = 12
        max_steps_per_restart = 60

        for r in range(n_restarts):
            if calls_made >= budget or time.time() >= deadline:
                break

            # Deterministic + random mix for diversity
            if r == 0:
                x0 = np.zeros(dim)
            elif r == 1:
                x0 = rng.uniform(-4, 4, dim)
            elif r == 2:
                x0 = rng.uniform(-3, 3, dim)
            elif r == 3:
                x0 = rng.uniform(-4.5, 4.5, dim)
            elif r == 4:
                x0 = rng.uniform(-2, 2, dim)
            else:
                x0 = rng.uniform(-5, 5, dim)

            fx0 = func(x0)
            calls_made += 1

            if fx0 < best_f:
                best_f, best_x = fx0, x0.copy()

            if calls_made >= budget or time.time() >= deadline:
                break

            remaining = budget - calls_made
            steps_allowed = min(max_steps_per_restart, remaining)
            if steps_allowed <= 0:
                break

            _, _, calls_made, best_f, best_x, _ = self._run_phase(
                func, rng, x0, fx0, calls_made, best_f, best_x,
                deadline, budget, dim, sigma_start, sigma_inc, sigma_dec,
                max_stall, steps_allowed, polish=False
            )

        # Phase 2: Polish around the best found point
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            polish_steps = min(30, remaining)
            if polish_steps > 0:
                self._run_phase(
                    func, rng, best_x, best_f, calls_made, best_f, best_x,
                    deadline, budget, dim,
                    sigma_start=0.3, sigma_inc=1.1, sigma_dec=0.75,
                    max_stall=10, max_steps=polish_steps, polish=True
                )

        # Phase 3: If we have budget left, do a few more small perturbations
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            for _ in range(min(remaining, 10)):
                if time.time() >= deadline:
                    break
                y = best_x + 0.15 * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls_made += 1
                if fy < best_f:
                    best_f, best_x = fy, y.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
