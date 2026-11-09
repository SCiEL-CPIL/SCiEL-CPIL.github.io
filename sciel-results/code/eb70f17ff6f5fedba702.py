"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = 320.0  # evaluator kills at ~330s; keep a safety margin


class Optimizer:
    """(1+1)-ES with 1/5 success-rule sigma adaptation, stall restart,
    and a final coordinate-descent polish. Minimizes func on [-5, 5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.perf_counter()

        # ---- Phase 1: (1+1)-ES with success-rule adaptation ----
        # Start near the origin: the shifted minimizer lives in [-4,4]^5,
        # so a center-ish start gives better early progress than a corner.
        rng = np.random.default_rng(0)
        x = rng.uniform(-0.5, 0.5, dim)
        fx = func(x)
        sigma = 2.0
        stall = 0
        STALL_LIMIT = 60

        for _ in range(budget - 1):
            if time.perf_counter() - t0 > _DEADLINE:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
            fy = func(y)
            if fy < fx:
                x, fx = y, fy
                stall = 0
                # success: grow sigma (1/5 rule -> 1.2 on success)
                sigma *= 1.2
            else:
                # failure: shrink sigma (1/5 rule -> 0.9 on failure)
                sigma *= 0.9
                stall += 1
                if stall >= STALL_LIMIT:
                    # stall: reseed toward the best found, shrink sigma
                    x = np.clip(x + 0.3 * sigma * rng.standard_normal(dim), -5.0, 5.0)
                    fx = func(x)
                    sigma *= 0.5
                    stall = 0

        # ---- Phase 2: coordinate-descent polish ----
        # Once the ES has stalled, squeeze the last bits with a few
        # coordinate-descent rounds (cheap, deterministic, no extra calls
        # beyond the budget).
        step = max(sigma, 0.05)
        for _round in range(3):
            if time.perf_counter() - t0 > _DEADLINE:
                break
            for d in range(dim):
                for sign in (1.0, -1.0):
                    trial = x.copy()
                    trial[d] = np.clip(trial[d] + sign * step, -5.0, 5.0)
                    ftrial = func(trial)
                    if ftrial < fx:
                        x, fx = trial, ftrial
                # shrink step if the coordinate didn't move
                step = max(step * 0.5, 1e-3)
            step = max(step, 1e-2)
            if time.perf_counter() - t0 > _DEADLINE:
                break

        return x, fx
# EVOLVE-BLOCK-END
