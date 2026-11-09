"""Deterministic multi-start grid + adaptive coordinate descent for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic multi-start grid search followed by adaptive coordinate descent."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        x_best = None
        f_best = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal x_best, f_best, calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline and calls_made < budget

        def coordinate_descent(x0, max_calls, init_step=0.5):
            """Adaptive coordinate descent with shrinking steps."""
            nonlocal x_best, f_best, calls_made
            x = np.array(x0, dtype=np.float64)
            f = f_best
            step = init_step
            no_improve = 0

            while calls_made < max_calls and time_ok() and step > 1e-10:
                improved = False
                for d in range(dim):
                    if calls_made >= max_calls or not time_ok():
                        break
                    for direction in (1, -1):
                        if calls_made >= max_calls or not time_ok():
                            break
                        trial = x.copy()
                        trial[d] += direction * step
                        trial = np.clip(trial, -5, 5)
                        f_trial = evaluate(trial)
                        if f_trial < f - 1e-12:
                            x = trial
                            f = f_trial
                            improved = True
                            if f < f_best:
                                f_best = f
                                x_best = x.copy()
                if improved:
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= 3:
                        step *= 0.5
                        no_improve = 0
                    else:
                        step *= 0.8

            return x, f

        def multi_start_grid_search():
            """Deterministic multi-start with structured grid + random points."""
            nonlocal x_best, f_best, calls_made

            # Phase 1: Coarse grid over [-4, 4]^dim
            # For dim=5, use 3^5=243 points on a coarse grid
            grid_values = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
            points = list(np.meshgrid(*([grid_values] * dim), indexing='ij'))
            points = np.array(points).reshape(dim, -1).T  # shape (N, dim)

            # Evaluate all grid points
            for i in range(len(points)):
                if not time_ok():
                    break
                evaluate(points[i])

            # Phase 2: Finer grid around best region
            if x_best is not None and time_ok():
                # Generate points around the best point with smaller spacing
                rng = np.random.default_rng(42)
                fine_step = 1.0
                for _ in range(5):
                    if not time_ok():
                        break
                    center = x_best.copy()
                    for _ in range(20):
                        if not time_ok():
                            break
                        x = center + rng.uniform(-fine_step, fine_step, dim)
                        x = np.clip(x, -5, 5)
                        evaluate(x)
                    fine_step *= 0.5

            # Phase 3: Random points to escape local optima
            if time_ok():
                rng = np.random.default_rng(123)
                for _ in range(50):
                    if not time_ok():
                        break
                    x = rng.uniform(-4.5, 4.5, dim)
                    evaluate(x)

        # Main strategy:
        # 1. Multi-start grid search to find a good basin
        # 2. Coordinate descent to refine
        # 3. More random restarts + coordinate descent if budget remains

        # Phase 1: Multi-start grid search (use ~60% of budget)
        grid_budget = int(budget * 0.55)
        saved_calls = calls_made
        multi_start_grid_search()
        grid_used = calls_made - saved_calls

        # Phase 2: Coordinate descent on best point
        if x_best is not None and time_ok():
            refine_budget = min(budget - calls_made, int(budget * 0.25))
            if refine_budget >= 5:
                coordinate_descent(x_best.copy(), calls_made + refine_budget, init_step=0.5)

        # Phase 3: If still budget, do more random restarts with coordinate descent
        if time_ok() and budget - calls_made >= 10:
            rng = np.random.default_rng(789)
            remaining = budget - calls_made
            while time_ok() and remaining >= 15:
                x = rng.uniform(-4.5, 4.5, dim)
                evaluate(x)
                remaining = budget - calls_made
                if remaining >= 15:
                    coordinate_descent(x_best.copy(), calls_made + min(15, remaining), init_step=0.3)
                    remaining = budget - calls_made

        # Phase 4: Final coordinate descent refinement
        if x_best is not None and time_ok() and budget - calls_made >= 5:
            coordinate_descent(x_best.copy(), budget, init_step=0.1)

        # Final safety: ensure we have a result
        if x_best is None:
            x_best = np.zeros(dim)
            f_best = evaluate(x_best)

        return x_best, f_best
# EVOLVE-BLOCK-END
