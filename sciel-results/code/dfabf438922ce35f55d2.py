"""Portfolio optimizer for BBOB f23 (Katsuura) on [-5,5]^5.

Strategy: LHS space-filling exploration for fast early AOCC progress,
then (1+1)-ES exploitation with adaptive sigma and periodic restarts
from the global best to escape Katsuura's dense local minima.
A wall-clock deadline guard returns the best result found with a safety margin.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: LHS exploration + (1+1)-ES with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before the 330s kill

        rng = np.random.default_rng(12345)
        lo, hi = -5.0, 5.0

        best_x = None
        best_f = np.inf
        calls = 0

        def eval_point(x):
            nonlocal calls, best_x, best_f
            calls += 1
            f = float(func(x))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # ---- Phase 1: LHS space-filling exploration (fast early progress) ----
        # Use a modest number of LHS points to cover the box well early.
        n_lhs = 30
        if budget > n_lhs:
            lhs = self._latin_hypercube(n_lhs, dim, rng)
            for x in lhs:
                if calls >= budget or time.time() > deadline:
                    break
                eval_point(x)

        # ---- Phase 2: (1+1)-ES exploitation with adaptive restarts ----
        # Start from the best point found so far.
        if best_x is None:
            best_x = rng.uniform(lo, hi, dim)
            best_f = eval_point(best_x)

        # Sigma adaptation state.
        sigma = 1.5
        # Restart based on no-improvement count
        no_improve = 0
        restart_threshold = 60
        restart_count = 0

        # Keep a local (1+1)-ES state that resets on restart.
        x = best_x.copy()
        fx = best_f

        while calls < budget and time.time() < deadline:
            # Propose a step.
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, lo, hi)
            fy = eval_point(y)

            if fy <= fx:
                # Improvement (or equal): move and grow sigma.
                x = y
                fx = fy
                sigma *= 1.3
                no_improve = 0
            else:
                # No improvement: shrink sigma.
                sigma *= 0.85
                no_improve += 1

            # Prevent sigma from becoming degenerate.
            sigma = min(max(sigma, 1e-3), 10.0)

            # Periodic restart: if stuck, return to global best, boost sigma to escape.
            if no_improve >= restart_threshold:
                restart_count += 1
                # On restart, use a larger sigma to jump out of the local basin.
                sigma = max(sigma, 3.0 + 1.0 * restart_count)
                x = best_x.copy()
                fx = best_f
                no_improve = 0
                # Increase restart threshold slightly to allow more exploration later
                restart_threshold = min(150, restart_threshold + 10)

        return best_x, best_f

    @staticmethod
    def _latin_hypercube(n, dim, rng):
        """Generate n Latin hypercube points in [lo, hi]^dim."""
        pts = np.empty((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            # For each row i, the value in dimension d is (perm[i] + U)/n
            u = rng.uniform(0, 1, n)
            vals = (perm + u) / n
            pts[:, d] = vals
        lo, hi = -5.0, 5.0
        pts = lo + (hi - lo) * pts
        return pts
# EVOLVE-BLOCK-END
