"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Restart-adaptive (1+1)-ES with basin-aware multi-start for Lunacek bi-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270.0  # safety margin before 330s kill
        rng = np.random.default_rng(0)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x = np.asarray(x, dtype=float)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def run_11es(x0, fx0, sigma0, max_calls):
            nonlocal calls, best_x, best_f, deadline
            n = dim
            x = np.array(x0, dtype=float)
            fx = fx0
            sigma = sigma0
            success = 0.0
            # 1/5 rule constants
            for _ in range(max_calls):
                if time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                    success = 1.0
                else:
                    sigma *= 0.87
                    success = 0.0
                if sigma < 1e-6:
                    break
            return x, fx

        # ---- Seed candidates: the two Lunacek basins + random ----
        # Basin A: near origin (the "x" attractor)
        # Basin B: near the shifted "d" region (d ~ 2.5 * x0-ish); place near (2.5,...)
        seeds = []
        # origin basin
        seeds.append(np.zeros(dim) + 0.5 * rng.standard_normal(dim))
        # shifted basin (Lunacek d is around 2.5 along the shift direction)
        for center_mag in (2.5, 3.0, 1.5):
            seeds.append(center_mag * np.ones(dim) + 0.5 * rng.standard_normal(dim))
            seeds.append(-center_mag * np.ones(dim) + 0.5 * rng.standard_normal(dim))
        # a few random seeds
        for _ in range(4):
            seeds.append(rng.uniform(-4.0, 4.0, dim))

        # Evaluate seeds
        seed_evals = []
        for s in seeds:
            if time.time() > deadline or calls >= budget:
                break
            fs = evaluate(s)
            seed_evals.append((s, fs))

        # Sort seeds by fitness, best first
        seed_evals.sort(key=lambda t: t[1])

        # ---- Restart-adaptive (1+1)-ES over the best seeds ----
        # Give the best seed the largest budget, then smaller budgets for others.
        # Stagnation: if a run makes no improvement for a window, restart from best seed.
        stagnation_window = 30
        # Budget allocation: best seed gets the bulk.
        remaining = budget - calls
        if remaining <= 0:
            return best_x, best_f

        # Allocate: 60% to best, rest split among next seeds
        alloc = []
        best_alloc = int(0.6 * remaining)
        alloc.append(best_alloc)
        rest = remaining - best_alloc
        n_rest = max(0, len(seed_evals) - 1)
        per_rest = rest // max(1, n_rest) if n_rest > 0 else 0
        for _ in range(n_rest):
            alloc.append(per_rest)

        # Track per-run best to detect stagnation
        for i, (sx, sf) in enumerate(seed_evals):
            if i >= len(alloc):
                break
            max_calls = alloc[i]
            if max_calls <= 0:
                break
            if time.time() > deadline:
                break

            # Run (1+1)-ES with stagnation restart from the current global best
            x = np.array(sx, dtype=float)
            fx = sf
            sigma = 2.0
            run_best_f = fx
            no_improve = 0
            steps = 0
            while steps < max_calls and time.time() < deadline and calls < budget:
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                steps += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                    if fx < run_best_f:
                        run_best_f = fx
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    sigma *= 0.87
                    no_improve += 1

                # Stagnation restart: jump to global best and re-explode
                if no_improve >= stagnation_window:
                    if best_x is not None:
                        x = best_x.copy() + 0.5 * sigma * rng.standard_normal(dim)
                        x = np.clip(x, -5.0, 5.0)
                        fx = evaluate(x)
                        steps += 1
                    sigma = max(sigma, 1.5)
                    no_improve = 0
                    if sigma < 1e-6:
                        break

                if sigma < 1e-6:
                    break

        # Final polish: a short (1+1)-ES from the global best with small sigma
        if best_x is not None and time.time() < deadline and calls < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            for _ in range(min(50, budget - calls)):
                if time.time() > deadline or calls >= budget:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.9

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = float(func(best_x))
            calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
