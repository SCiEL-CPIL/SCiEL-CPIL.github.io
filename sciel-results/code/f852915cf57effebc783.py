"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Linear gradient detection: use simplex to find gradient, jump to exact optimal corner."""

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

        # Phase 1: Estimate gradient via forward differences from origin
        # For linear f(x) = c + g·x, we can determine g from 6 evaluations:
        # f(0) and f(e_i * h) for i=0..dim-1
        h = 4.0
        f0 = fx_best
        grad = np.zeros(dim)
        
        for i in range(dim):
            if not time_ok() or calls >= budget:
                break
            x_probe = np.zeros(dim)
            x_probe[i] = h
            f_probe = func(x_probe)
            calls += 1
            if f_probe < fx_best:
                x_best = x_probe.copy()
                fx_best = f_probe
            # For linear function: f(h*e_i) - f(0) = g_i * h
            grad[i] = (f_probe - f0) / h

        # Phase 2: Jump to the optimal corner
        # For linear f(x) = c + g·x on [-5,5]^d, min is at x_i = -5*sign(g_i)
        # i.e., x_i = 5 if g_i < 0, x_i = -5 if g_i > 0
        x_corner = np.where(grad < 0, 5.0, -5.0)
        
        if time_ok() and calls < budget:
            fx_corner = func(x_corner)
            calls += 1
            if fx_corner < fx_best:
                x_best = x_corner.copy()
                fx_best = fx_corner

        # Phase 3: Verify by checking the opposite corner (safety net)
        x_corner_opp = np.where(grad < 0, -5.0, 5.0)
        if time_ok() and calls < budget:
            fx_opp = func(x_corner_opp)
            calls += 1
            if fx_opp < fx_best:
                x_best = x_corner_opp.copy()
                fx_best = fx_opp

        # Phase 4: Use remaining budget for coordinate descent polish
        # This handles any slight non-linearity or numerical issues
        x = x_best.copy()
        fx = fx_best

        remaining = budget - calls
        if remaining > 0 and time_ok():
            # Adaptive step coordinate descent
            step_sizes = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
            for step in step_sizes:
                if not time_ok() or calls >= budget:
                    break
                # Do as many passes as budget allows
                max_passes = max(1, (budget - calls) // (dim * 2))
                for _ in range(min(max_passes, 3)):
                    if not time_ok() or calls >= budget:
                        break
                    for i in range(dim):
                        if not time_ok() or calls >= budget:
                            break
                        # Try positive direction
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
                        # Try negative direction
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
