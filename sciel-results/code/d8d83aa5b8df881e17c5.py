"""Rewritten optimizer for BBOB-style f14 (Different powers)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Deterministic multi-scale coordinate descent with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin under 330s wall-clock
        dim = self.dim
        budget = self.budget

        # ---- Phase A: deterministic coarse multi-scale coordinate descent ----
        # Start at origin (minimizer is known to lie in [-4,4]^5).
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1

        def time_ok():
            return time.time() < deadline

        def try_step(x, fx, step, direction):
            """Try a move along direction*step; return (x, fx, improved, used)."""
            nonlocal best_x, best_f, calls
            y = np.clip(x + step * direction, -5.0, 5.0)
            fy = func(y)
            calls += 1
            if fy < fx:
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                return y, fy, True
            return x, fx, False

        # Geometrically shrinking strides: 4.0 -> ... -> small
        stride = 4.0
        # Coordinate descent sweeps over each axis with +/- step.
        min_stride = 1e-4
        phase_a_budget = int(0.45 * budget)
        a_calls = 0
        while a_calls < phase_a_budget and time_ok():
            improved = False
            for i in range(dim):
                if a_calls >= phase_a_budget or not time_ok():
                    break
                for sgn in (1.0, -1.0):
                    if a_calls >= phase_a_budget or not time_ok():
                        break
                    d = np.zeros(dim)
                    d[i] = sgn
                    x, fx, imp = try_step(x, fx, stride, d)
                    a_calls += 1
                    improved = improved or imp
            if not improved:
                # No improvement this sweep: shrink stride.
                stride *= 0.5
                if stride < min_stride:
                    break
            else:
                # Keep stride while still making progress (refine in place).
                pass

        # ---- Phase B: fine coordinate descent + light random restarts ----
        # Restore to best found and refine with small adaptive step.
        x = best_x.copy()
        fx = best_f
        step = 0.5
        min_step = 1e-5
        rng = np.random.default_rng(12345)
        b_calls = 0
        max_b = budget - calls
        # Fine deterministic coordinate descent first.
        while b_calls < max_b and time_ok():
            improved = False
            for i in range(dim):
                if b_calls >= max_b or not time_ok():
                    break
                for sgn in (1.0, -1.0):
                    if b_calls >= max_b or not time_ok():
                        break
                    d = np.zeros(dim)
                    d[i] = sgn
                    x, fx, imp = try_step(x, fx, step, d)
                    b_calls += 1
                    improved = improved or imp
            if not improved:
                step *= 0.5
                if step < min_step:
                    break

        # Light random-restart refinement to escape plateaus, keep global best.
        restarts = 6
        per_restart = max(5, (budget - calls) // (restarts + 1))
        for r in range(restarts):
            if calls >= budget or not time_ok():
                break
            # Perturb around the current best.
            x = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            step = 0.25
            local_calls = 0
            while local_calls < per_restart and calls < budget and time_ok():
                improved = False
                for i in range(dim):
                    if local_calls >= per_restart or calls >= budget or not time_ok():
                        break
                    for sgn in (1.0, -1.0):
                        if local_calls >= per_restart or calls >= budget or not time_ok():
                            break
                        d = np.zeros(dim)
                        d[i] = sgn
                        x, fx, imp = try_step(x, fx, step, d)
                        local_calls += 1
                        improved = improved or imp
                if not improved:
                    step *= 0.5
                    if step < min_step:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
