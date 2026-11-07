"""Rewritten DFCD-based optimizer for BBOB f24 (Lunacek bi-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Derivative-Free Coordinate Descent with adaptive step scheduling
    tuned for the bi-modal Lunacek bi-Rastrigin function in 5D."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        best_x = None
        best_fx = float('inf')
        calls_made = 0

        def _eval(x):
            nonlocal calls_made, best_x, best_fx
            fx = float(func(x))
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = np.array(x, dtype=float, copy=True)
            return fx

        # ----- Phase 1: coarse multi-start scan to locate the global basin -----
        # The Lunacek function has a global minimum near x0 and a local minimum
        # near -1/0.81*x0.  We probe several representative regions.
        starts = [
            np.zeros(dim),
            np.full(dim, 1.5),
            np.full(dim, -1.5),
            np.full(dim, 3.0),
            np.full(dim, -3.0),
            np.array([2.0, -2.0, 2.0, -2.0, 2.0][:dim]),
            np.array([-2.0, 2.0, -2.0, 2.0, -2.0][:dim]),
            np.array([0.5, 0.5, 0.5, 0.5, 0.5][:dim]),
            np.array([-0.5, -0.5, -0.5, -0.5, -0.5][:dim]),
        ]

        for s in starts:
            if calls_made >= budget - 20:
                break
            _eval(s)

        # ----- Phase 2: adaptive coordinate descent from the current best -----
        # Geometrically shrinking steps, cycling dimensions, with a small
        # tolerance to stop when no improvement is found at a given scale.
        base_steps = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625,
                      0.0078125, 0.00390625, 0.001953125, 0.0009765625]

        if best_x is not None:
            x = best_x.copy()
            fx = best_fx

            for step in base_steps:
                if calls_made >= budget - 5:
                    break
                improved = False
                # Two passes over dimensions per step for better convergence
                for _pass in range(2):
                    for i in range(dim):
                        if calls_made >= budget - 5:
                            break
                        # positive
                        x_try = x.copy()
                        x_try[i] = min(5.0, x_try[i] + step)
                        if abs(x_try[i] - x[i]) > 1e-12:
                            fx_try = _eval(x_try)
                            if fx_try < fx:
                                x = x_try
                                fx = fx_try
                                improved = True
                                continue
                        # negative
                        x_try = x.copy()
                        x_try[i] = max(-5.0, x_try[i] - step)
                        if abs(x_try[i] - x[i]) > 1e-12:
                            fx_try = _eval(x_try)
                            if fx_try < fx:
                                x = x_try
                                fx = fx_try
                                improved = True
                if not improved and step < 0.01:
                    break

        # ----- Phase 3: fine polish with very small steps -----
        if best_x is not None and calls_made < budget - 3:
            x = best_x.copy()
            fx = best_fx
            fine_steps = [0.0005, 0.00025, 0.000125, 0.0000625]
            for step in fine_steps:
                if calls_made >= budget - 3:
                    break
                improved = False
                for i in range(dim):
                    if calls_made >= budget - 3:
                        break
                    x_try = x.copy()
                    x_try[i] = min(5.0, x_try[i] + step)
                    if abs(x_try[i] - x[i]) > 1e-12:
                        fx_try = _eval(x_try)
                        if fx_try < fx:
                            x = x_try
                            fx = fx_try
                            improved = True
                            continue
                    x_try = x.copy()
                    x_try[i] = max(-5.0, x_try[i] - step)
                    if abs(x_try[i] - x[i]) > 1e-12:
                        fx_try = _eval(x_try)
                        if fx_try < fx:
                            x = x_try
                            fx = fx_try
                            improved = True
                if not improved:
                    break

        return best_x, best_fx

# EVOLVE-BLOCK-END
