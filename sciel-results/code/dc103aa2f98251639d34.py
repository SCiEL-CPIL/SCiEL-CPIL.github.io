"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive sigma, restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _run_one_restart(self, func, rng, x0, fx0, budget_for_restart, deadline):
        """Run a single (1+1)-ES restart. Returns (best_x, best_fx, calls_used)."""
        x = x0.copy()
        fx = fx0
        best_x = x.copy()
        best_fx = fx
        sigma = 1.5
        calls = 1  # fx0 already evaluated
        stall = 0
        stall_limit = 30

        while calls < budget_for_restart:
            if time.time() > deadline:
                break

            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x = y
                fx = fy
                if fy < best_fx:
                    best_x = x.copy()
                    best_fx = fy
                sigma *= 1.1
                stall = 0
            else:
                sigma *= 0.9
                stall += 1

            if stall >= stall_limit:
                # Nudge: take a random step to escape local basin
                x = np.clip(x + 2.0 * rng.standard_normal(self.dim), -5, 5)
                fx = func(x)
                calls += 1
                if fx < best_fx:
                    best_x = x.copy()
                    best_fx = fx
                sigma = 1.5
                stall = 0

        return best_x, best_fx, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s kill
        rng = np.random.default_rng(42)

        total_budget = self.budget
        dim = self.dim

        best_x = None
        best_fx = np.inf
        total_calls = 0

        # Phase 1: Initial exploration with a few random restarts
        # Use a portion of budget for multiple short restarts to find promising basins
        n_initial_restarts = 5
        budget_per_restart = max(30, total_budget // (n_initial_restarts + 5))

        for i in range(n_initial_restarts):
            if total_calls >= total_budget or time.time() > deadline:
                break

            x0 = rng.uniform(-5, 5, dim)
            fx0 = func(x0)
            total_calls += 1

            if fx0 < best_fx:
                best_x = x0.copy()
                best_fx = fx0

            remaining_for_this = min(budget_per_restart - 1, total_budget - total_calls)
            if remaining_for_this <= 0:
                continue

            bx, bfx, calls_used = self._run_one_restart(
                func, rng, x0, fx0, remaining_for_this + 1, deadline
            )
            total_calls += calls_used - 1  # subtract the initial eval already counted

            if bfx < best_fx:
                best_x = bx
                best_fx = bfx

        # Phase 2: Continue from the best point found, using remaining budget
        if best_x is not None and total_calls < total_budget:
            remaining = total_budget - total_calls
            if remaining > 0:
                bx, bfx, calls_used = self._run_one_restart(
                    func, rng, best_x, best_fx, remaining + 1, deadline
                )
                total_calls += calls_used - 1
                if bfx < best_fx:
                    best_x = bx
                    best_fx = bfx

        # Ensure we don't exceed budget
        if best_x is None:
            x0 = rng.uniform(-5, 5, dim)
            fx0 = func(x0)
            best_x = x0
            best_fx = fx0

        return best_x, best_fx
# EVOLVE-BLOCK-END
