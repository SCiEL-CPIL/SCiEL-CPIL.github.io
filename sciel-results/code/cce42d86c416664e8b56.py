"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio optimizer: coordinate descent + CMA-ES style search with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        # Counters
        calls = [0]

        def safe_func(x):
            if calls[0] >= budget:
                raise StopIteration
            calls[0] += 1
            return float(func(np.asarray(x, dtype=np.float64)))

        # ---- Phase 1: Multi-start coordinate descent with random perturbations ----
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        # Initial diverse starts
        starts = []
        starts.append(np.array([0.0] * dim))
        starts.append(np.array([2.0] * dim))
        starts.append(np.array([-2.0] * dim))
        starts.append(rng.uniform(-4, 4, dim))
        starts.append(rng.uniform(-4, 4, dim))

        for s in starts:
            if calls[0] >= budget:
                break
            x = np.clip(s, -5, 5)
            try:
                fx = safe_func(x)
            except StopIteration:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Coordinate descent refinement
        step = 1.0
        for _ in range(20):
            if calls[0] >= budget:
                break
            if time.time() > deadline:
                break
            improved = False
            for d in range(dim):
                if time.time() > deadline:
                    break
                for sign in (1, -1):
                    if calls[0] >= budget:
                        break
                    x2 = best_x.copy()
                    x2[d] = np.clip(best_x[d] + sign * step, -5, 5)
                    try:
                        f2 = safe_func(x2)
                    except StopIteration:
                        break
                    if f2 < best_f:
                        best_f = f2
                        best_x = x2
                        improved = True
                if time.time() > deadline:
                    break
            if improved:
                step *= 0.8
            else:
                step *= 0.6
            if step < 1e-5:
                break

        # ---- Phase 2: CMA-ES-like evolution strategy ----
        # Use the best point as center, adapt step size
        center = best_x.copy()
        cf = best_f
        sigma = 0.5
        lam = 4  # population size for small dim

        while calls[0] < budget and time.time() < deadline:
            # Generate population
            improved_any = False
            for i in range(lam):
                if calls[0] >= budget or time.time() > deadline:
                    break
                child = np.clip(center + sigma * rng.standard_normal(dim), -5, 5)
                try:
                    cf_c = safe_func(child)
                except StopIteration:
                    break
                if cf_c < cf:
                    cf = cf_c
                    center = child
                    improved_any = True

            # Adapt sigma
            if improved_any:
                sigma *= 1.2
            else:
                sigma *= 0.8

            if sigma < 1e-7:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
