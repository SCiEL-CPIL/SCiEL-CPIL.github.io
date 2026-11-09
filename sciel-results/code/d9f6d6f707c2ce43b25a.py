"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start (1+1)-ES with 1/5 success-rule step-size adaptation for
    BBOB f18 (Schaffers F7 ill-conditioned), dim=5, box [-5,5]^5, budget 500.

    Strategy:
      - Small random sampling to locate a promising basin.
      - Several independent (1+1)-ES runs, each starting from the current
        global best with a fresh (larger) sigma, to escape narrow minima.
      - 1/5 rule adapts sigma toward the optimal success rate.
      - Time guard with a safety margin; a final polish loop spends any
        remaining calls on a tight local search around the best point.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _es_run(self, func, rng, x, fx, sigma, n_evals, best, deadline):
        """
        Run one (1+1)-ES segment with 1/5 success rule.
        Returns (x, fx, sigma, n_evals, best).
        """
        dim = self.dim
        # 1/5 success rule constants
        c_sigma = 1.0 / (dim + 4.0)
        h = 0.84 + 2.0 / (dim + 1.0)
        sigma_exp = 0.0
        steps_since_restart = 0

        while n_evals < self.budget:
            if time.time() > deadline:
                break

            # Candidate
            y = np.clip(x + sigma * (1.0 + sigma_exp) * rng.standard_normal(dim), -5.0, 5.0)
            fy = func(y)
            n_evals += 1
            steps_since_restart += 1

            if fy < fx:
                # Success
                x, fx = y, fy
                if fx < best[0]:
                    best[0] = fx
                    best[1] = x.copy()
                sigma_exp = (1.0 - c_sigma) * sigma_exp + c_sigma * h
            else:
                # Failure
                sigma_exp = (1.0 - c_sigma) * sigma_exp

            sigma = np.exp(sigma_exp)
            sigma = float(np.clip(sigma, 1e-5, 8.0))

        return x, fx, sigma, n_evals, best

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget
        n_evals = 0

        best = [np.inf, np.zeros(dim)]

        def record(x, fx):
            nonlocal n_evals
            if fx < best[0]:
                best[0] = fx
                best[1] = x.copy()

        # ---- Phase 1: Random sampling to find a promising basin ----
        sample_budget = max(8, int(0.12 * budget))  # ~60 calls
        for _ in range(sample_budget):
            if n_evals >= budget:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            fx = func(x)
            n_evals += 1
            record(x, fx)

        # ---- Phase 2: Multi-start (1+1)-ES ----
        # Split remaining budget into segments. More, shorter runs help escape
        # the narrow minima of F7; each restarts from the current global best.
        remaining = budget - n_evals
        n_runs = 6
        seg = max(5, remaining // n_runs)

        for run in range(n_runs):
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break

            # Start from the current global best; fresh larger sigma for exploration
            x = best[1].copy()
            fx = best[0]
            # sigma shrinks over runs to allow progressively finer search
            sigma0 = 3.0 * (0.72 ** run)
            sigma0 = max(sigma0, 0.2)

            x, fx, sigma, n_evals, best = self._es_run(
                func, rng, x, fx, sigma0, n_evals, best, deadline
            )

        # ---- Phase 3: Final polish with any leftover calls ----
        if n_evals < budget and time.time() < deadline:
            x = best[1].copy()
            fx = best[0]
            sigma = 0.05  # very small for fine local search
            x, fx, sigma, n_evals, best = self._es_run(
                func, rng, x, fx, sigma, n_evals, best, deadline
            )

        return best[1], best[0]
# EVOLVE-BLOCK-END
