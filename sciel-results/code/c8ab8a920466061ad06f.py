"""Optimiser for BBOB f4 (Büche-Rastrigin) with a deterministic multi-scale coordinate descent."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-scale coordinate descent for 5-D Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before the 300 s wall-clock limit

        # --- Phase 0: initial coarse scan to get a reasonable starting point ---
        # Evaluate a small grid of candidate points to locate a promising region.
        # The minimiser is known to lie in [-4, 4]^5, so we scan that box.
        # We use a coarse grid of 3 points per dimension: -4, 0, 4 → 3^5 = 243 points.
        # This fits within budget and gives early progress quickly.

        x = np.zeros(self.dim)
        fx = np.inf

        # Coarse grid: 3 levels per dimension over [-4, 4]
        levels = np.array([-4.0, 0.0, 4.0])
        calls_used = 0

        # Enumerate all grid points
        for i0 in range(3):
            for i1 in range(3):
                for i2 in range(3):
                    for i3 in range(3):
                        for i4 in range(3):
                            if calls_used >= self.budget:
                                break
                            if time.time() > deadline:
                                return x, fx
                            pt = np.array([levels[i0], levels[i1], levels[i2],
                                           levels[i3], levels[i4]], dtype=np.float64)
                            fval = float(func(pt))
                            calls_used += 1
                            if fval < fx:
                                fx = fval
                                x = pt.copy()

        # --- Phase 1: multi-scale coordinate descent ---
        # Geometrically shrinking step sizes, cycling through dimensions.
        # Start with a large step and shrink by a factor each full cycle.

        # Step sizes: start at 4.0 (covers the box) and shrink by 0.5 each cycle
        step = 4.0
        shrink = 0.5
        min_step = 1e-6

        # For each step size, do coordinate descent: for each dim, try +/- step
        # Use a fixed number of passes per step size to use budget efficiently.

        while step >= min_step:
            if time.time() > deadline:
                return x, fx
            if calls_used >= self.budget:
                break

            # Number of coordinate-descent passes at this scale:
            # Use remaining budget wisely; at least 1 pass, more if budget allows.
            # Each pass costs 2 * dim evaluations (try +step and -step for each dim).
            evals_per_pass = 2 * self.dim
            remaining = self.budget - calls_used
            passes = max(1, min(10, remaining // evals_per_pass))

            for _pass in range(passes):
                if time.time() > deadline:
                    return x, fx
                if calls_used >= self.budget:
                    break

                improved = False
                for d in range(self.dim):
                    if calls_used >= self.budget:
                        break
                    if time.time() > deadline:
                        return x, fx

                    # Try +step in dimension d
                    x_plus = x.copy()
                    x_plus[d] += step
                    x_plus = np.clip(x_plus, -5.0, 5.0)
                    f_plus = float(func(x_plus))
                    calls_used += 1

                    if f_plus < fx:
                        x = x_plus
                        fx = f_plus
                        improved = True
                    else:
                        # Try -step in dimension d
                        x_minus = x.copy()
                        x_minus[d] -= step
                        x_minus = np.clip(x_minus, -5.0, 5.0)
                        f_minus = float(func(x_minus))
                        calls_used += 1

                        if f_minus < fx:
                            x = x_minus
                            fx = f_minus
                            improved = True

                if not improved:
                    # No improvement in this pass; shrink step
                    step *= shrink
                    break

            step *= shrink

        # --- Phase 2: fine-tuning with very small steps ---
        # Use remaining budget for a final fine coordinate descent
        fine_step = 1e-4
        while fine_step >= 1e-8:
            if time.time() > deadline:
                return x, fx
            if calls_used >= self.budget:
                break

            evals_per_pass = 2 * self.dim
            remaining = self.budget - calls_used
            if remaining < evals_per_pass:
                break

            passes = max(1, min(5, remaining // evals_per_pass))
            for _pass in range(passes):
                if time.time() > deadline or calls_used >= self.budget:
                    break
                improved = False
                for d in range(self.dim):
                    if calls_used >= self.budget:
                        break
                    if time.time() > deadline:
                        return x, fx

                    x_plus = x.copy()
                    x_plus[d] += fine_step
                    x_plus = np.clip(x_plus, -5.0, 5.0)
                    f_plus = float(func(x_plus))
                    calls_used += 1

                    if f_plus < fx:
                        x = x_plus
                        fx = f_plus
                        improved = True
                    else:
                        x_minus = x.copy()
                        x_minus[d] -= fine_step
                        x_minus = np.clip(x_minus, -5.0, 5.0)
                        f_minus = float(func(x_minus))
                        calls_used += 1

                        if f_minus < fx:
                            x = x_minus
                            fx = f_minus
                            improved = True

                if not improved:
                    break

            fine_step *= 0.5

        return x, fx
# EVOLVE-BLOCK-END
