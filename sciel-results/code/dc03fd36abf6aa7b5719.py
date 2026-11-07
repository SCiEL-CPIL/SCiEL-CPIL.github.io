"""Hybrid global-local optimizer for BBOB f7 (Step ellipsoidal)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Portfolio: coarse grid + random restarts + adaptive (1+1)-ES refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 290.0  # 40s safety margin before 330s kill
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        call_count = 0

        def try_point(pt):
            nonlocal best_x, best_f, call_count
            if call_count >= budget:
                return
            f = func(pt)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        # Phase 1: coarse grid search to find a good basin.
        # The step ellipsoidal has a large flat region; locating the center
        # (the minimizer) is the key move. A 2^5 grid gives 32 points.
        grid_vals = np.linspace(-4.5, 4.5, 2)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        for m in range(2):
                            if time.time() > deadline:
                                return best_x, best_f
                            pt = np.array([grid_vals[i], grid_vals[j], grid_vals[k],
                                          grid_vals[l], grid_vals[m]], dtype=float)
                            try_point(pt)

        # Phase 2: random restarts to explore other basins briefly.
        for _ in range(15):
            if time.time() > deadline or call_count >= budget:
                break
            pt = rng.uniform(-4.5, 4.5, dim)
            try_point(pt)

        # Phase 3: adaptive (1+1)-ES with 1/5 success rule.
        # The step ellipsoidal is unimodal with a flat bottom, so a
        # well-tuned (1+1)-ES should quickly converge to the minimizer.
        x = best_x.copy()
        fx = best_f
        sigma = 2.0
        successes = 0
        window = 15
        trials_in_window = 0

        while call_count < budget:
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            try_point(y)
            if call_count >= budget:
                break
            # Re-fetch fx/best_x in case try_point updated them
            if y is best_x:
                fx = best_f
            else:
                # y was not better; fx unchanged
                pass
            trials_in_window += 1
            if best_x is not None and np.array_equal(y, best_x):
                successes += 1
            if trials_in_window >= window:
                rate = successes / trials_in_window
                if rate > 0.35:
                    sigma *= 1.15
                elif rate < 0.15:
                    sigma *= 0.75
                else:
                    sigma *= 0.98
                successes = 0
                trials_in_window = 0
            sigma = max(sigma, 1e-7)

        return best_x, best_f
# EVOLVE-BLOCK-END
