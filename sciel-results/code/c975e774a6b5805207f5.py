"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Portfolio optimizer for BBOB f18 (Schaffers F7 ill-conditioned).
    Strategy:
      - Random sampling phase to get several diverse starting points.
      - Run multiple (1+1)-ES runs from the best samples, each with a
        sub-budget and adaptive step size (1/5 success rule variant).
      - Keep a global best across all runs.
      - Explicit wall-clock deadline guard (safety margin before 330 s).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 320.0  # safety margin vs 330 s wall-clock
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(12345)

        calls = 0

        def safe_call(x):
            nonlocal calls
            calls += 1
            return func(x)

        # ---- Phase 1: random sampling to seed the portfolio ----
        n_samples = min(20, budget // 5)
        samples = []
        for _ in range(n_samples):
            if calls >= budget or time.time() >= deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            samples.append((fx, x))

        if not samples:
            return 0.0

        samples.sort(key=lambda t: t[0])
        best_fx, best_x = samples[0]

        # ---- Phase 2: multiple (1+1)-ES runs from best seeds ----
        # Allocate remaining budget across a few runs.
        remaining = budget - calls
        if remaining <= 0 or time.time() >= deadline:
            return best_fx

        # Use the top few distinct seeds (dedup by value proximity)
        seeds = []
        for fx, x in samples[:5]:
            if all(np.linalg.norm(x - sx) > 0.1 for _, sx in seeds):
                seeds.append((fx, x))
        if not seeds:
            seeds = [samples[0]]

        n_runs = len(seeds)
        per_run_budget = max(1, remaining // n_runs)

        for fx0, x0 in seeds:
            x = x0.copy()
            fx = fx0
            sigma = 1.0
            # Adaptive sigma with 1/5 rule
            successes = 0
            total = 0
            for _ in range(per_run_budget):
                if calls >= budget or time.time() >= deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                total += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    sigma *= 1.1
                    if fx < best_fx:
                        best_fx, best_x = fx, x
                else:
                    sigma *= 0.9
                # 1/5 success rule adaptation every 1+dim trials
                if total >= 1 + dim:
                    rate = successes / total
                    if rate > 0.3:
                        sigma *= 1.2
                    elif rate < 0.1:
                        sigma *= 0.7
                    successes = 0
                    total = 0
                # Keep sigma in a sane range
                sigma = np.clip(sigma, 1e-4, 10.0)

            if time.time() >= deadline or calls >= budget:
                break

        return best_fx
# EVOLVE-BLOCK-END
