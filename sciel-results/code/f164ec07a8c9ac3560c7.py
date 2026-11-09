"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Diverse restart portfolio with adaptive fine local search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0

        def time_ok():
            return time.time() < deadline

        calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        best_x = np.zeros(dim)
        best_f = np.inf

        def try_point(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or not time_ok():
                return
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1: Quick diverse sampling to find promising region
        # Stratified sampling in [-4, 4]^dim
        n_sample = 30
        for i in range(n_sample):
            if calls >= budget or not time_ok():
                break
            # Use Latin hypercube-like stratification
            x = rng.uniform(-4, 4, dim)
            try_point(x)

        # Phase 2: Short (1+1)-ES from top 3 diverse points
        # We track the best few points from sampling
        # Actually, let's just do (1+1)-ES from the current best and a few alternates
        
        # Better: do short ES from several diverse starts
        # Pick 5 diverse starting points from our samples
        # For simplicity, just do ES from best and a few random
        
        # Phase 2: (1+1)-ES with adaptive sigma from best point
        # But first, let's try a few more diverse starts with short ES
        
        es_starts = [best_x.copy()]
        # Add 4 more diverse starts
        for _ in range(4):
            es_starts.append(rng.uniform(-4, 4, dim))

        for start in es_starts:
            if calls >= budget or not time_ok():
                break
            x = np.clip(start, -5, 5).copy()
            fx = best_f if np.allclose(x, best_x) else func(x)
            if not np.allclose(x, best_x):
                calls += 1
                if fx < best_f:
                    best_f, best_x = fx, x.copy()

            sigma = 1.0
            short_budget = 20
            for _ in range(short_budget):
                if calls >= budget or not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.15
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma *= 0.85

        # Phase 3: Fine-grained local search
        remaining = budget - calls
        if remaining > 0 and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.2
            for _ in range(remaining):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma *= 0.93
                    if sigma < 1e-5:
                        sigma = 1e-5

        return best_x, best_f
# EVOLVE-BLOCK-END
