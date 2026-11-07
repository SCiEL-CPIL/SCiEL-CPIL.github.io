"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Corner-detection optimizer for linear slope: identify optimal corner signs, then polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0

        def time_ok():
            return time.time() < deadline

        x_best = np.zeros(dim)
        fx_best = func(x_best)
        calls = 1

        # Phase 1: Determine optimal sign per dimension by probing near boundaries
        # For a linear function f(x) = c + g·(x - x*), the minimum is at a corner.
        # We probe: for each dimension, compare f at +4 and -4 (relative to center 0)
        # to determine which direction decreases f.
        probe_pos = np.zeros(dim)
        probe_neg = np.zeros(dim)
        f_pos = np.full(dim, np.inf)
        f_neg = np.full(dim, np.inf)

        for i in range(dim):
            if not time_ok() or calls >= budget:
                break
            probe_pos[i] = 4.0
            probe_neg[i] = -4.0

        # Evaluate f at all-positive and all-negative probes
        if time_ok() and calls < budget:
            f_all_pos = func(probe_pos.copy())
            calls += 1
            if f_all_pos < fx_best:
                x_best = probe_pos.copy()
                fx_best = f_all_pos

        if time_ok() and calls < budget:
            f_all_neg = func(probe_neg.copy())
            calls += 1
            if f_all_neg < fx_best:
                x_best = probe_neg.copy()
                fx_best = f_all_neg

        # Phase 2: Per-dimension sign determination using paired evaluations
        # For each dimension i, evaluate f with x[i]=+4 (others at 0) vs x[i]=-4
        # This gives us the sign of the gradient component.
        signs = np.zeros(dim)
        for i in range(dim):
            if not time_ok() or calls >= budget:
                break
            x_p = np.zeros(dim)
            x_p[i] = 4.0
            x_n = np.zeros(dim)
            x_n[i] = -4.0
            fp = func(x_p)
            calls += 1
            if not time_ok() or calls >= budget:
                break
            fn = func(x_n)
            calls += 1
            # If f(+4) < f(-4), gradient is negative → min is at +5
            # If f(-4) < f(+4), gradient is positive → min is at -5
            if fp < fn:
                signs[i] = 1.0   # go positive
            else:
                signs[i] = -1.0  # go negative

        # Phase 3: Jump to the predicted corner
        x_corner = np.where(signs > 0, 5.0, -5.0)
        if time_ok() and calls < budget:
            fx_corner = func(x_corner)
            calls += 1
            if fx_corner < fx_best:
                x_best = x_corner.copy()
                fx_best = fx_corner

        # Phase 4: Try the opposite corner too (in case sign determination was noisy)
        x_corner_opp = np.where(signs > 0, -5.0, 5.0)
        if time_ok() and calls < budget:
            fx_opp = func(x_corner_opp)
            calls += 1
            if fx_opp < fx_best:
                x_best = x_corner_opp.copy()
                fx_best = fx_opp

        # Phase 5: Coordinate descent polish - refine each coordinate
        # For a truly linear function, the optimum is exactly at a corner,
        # but numerical precision or slight non-linearity may benefit from polishing.
        x = x_best.copy()
        fx = fx_best

        # Use remaining budget for iterative coordinate descent with adaptive step
        remaining = budget - calls
        if remaining > 0 and time_ok():
            # Multi-pass coordinate descent
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            for step in step_sizes:
                if not time_ok() or calls >= budget:
                    break
                for _ in range(min(2, max(1, (budget - calls) // (dim * 2)))):
                    if not time_ok() or calls >= budget:
                        break
                    for i in range(dim):
                        if not time_ok() or calls >= budget:
                            break
                        # Try moving in positive direction
                        x_trial = x.copy()
                        x_trial[i] = min(5.0, x[i] + step)
                        ft = func(x_trial)
                        calls += 1
                        if ft < fx:
                            x = x_trial.copy()
                            fx = ft
                            if fx < fx_best:
                                x_best = x.copy()
                                fx_best = fx
                        # Try moving in negative direction
                        if not time_ok() or calls >= budget:
                            break
                        x_trial = x.copy()
                        x_trial[i] = max(-5.0, x[i] - step)
                        ft = func(x_trial)
                        calls += 1
                        if ft < fx:
                            x = x_trial.copy()
                            fx = ft
                            if fx < fx_best:
                                x_best = x.copy()
                                fx_best = fx

        return x_best, fx_best
# EVOLVE-BLOCK-END
