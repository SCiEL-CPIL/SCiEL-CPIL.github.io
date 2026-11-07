"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive per-axis optimizer exploiting ridge anisotropy."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            calls_made += 1
            return val

        # Phase 1: Global search with population and per-axis adaptive steps
        # Use a small population to explore the landscape
        pop_size = min(10, max(4, budget // 50))
        x = rng.uniform(-5, 5, n)
        fx = safe_call(x)
        if fx < best_f:
            best_f, best_x = fx, x.copy()

        # Per-axis step sizes, initialized to explore the box
        sigma = np.full(n, 2.0)

        # Global exploration phase: ~40% of budget
        global_budget = int(budget * 0.4)
        calls_in_phase = 1  # initial call

        while calls_in_phase < global_budget and time.time() < deadline:
            # Generate a candidate by perturbing the current best
            direction = rng.standard_normal(n)
            y = best_x + sigma * direction
            fy = safe_call(y)
            calls_in_phase += 1

            if fy < best_f:
                best_f, best_x = fy, y.copy()
                # Adapt step sizes: increase for axes that helped
                for i in range(n):
                    if abs(direction[i]) > 0.1:
                        sigma[i] *= 1.2
            else:
                # Decrease step sizes for axes that didn't help
                for i in range(n):
                    if abs(direction[i]) > 0.1:
                        sigma[i] *= 0.85

            # Periodically try a fresh random point to escape local minima
            if calls_in_phase % 20 == 0:
                x_try = rng.uniform(-5, 5, n)
                fy_try = safe_call(x_try)
                calls_in_phase += 1
                if fy_try < best_f:
                    best_f, best_x = fy_try, x_try.copy()
                    sigma = np.full(n, 2.0)

        # Phase 2: Coordinate descent with per-axis adaptive steps
        # This exploits the ridge structure: narrow directions converge faster
        # with axis-aligned steps
        cd_budget = int(budget * 0.45)
        calls_in_phase2 = 0

        # Initialize per-axis step sizes for coordinate descent
        axis_sigma = np.full(n, 0.5)
        x_cd = best_x.copy()
        f_cd = best_f

        while calls_in_phase2 < cd_budget and time.time() < deadline:
            improved = False
            for i in range(n):
                if calls_in_phase2 >= cd_budget or time.time() > deadline:
                    break
                # Try positive step
                x_try = x_cd.copy()
                x_try[i] += axis_sigma[i]
                f_try = safe_call(x_try)
                calls_in_phase2 += 1

                if f_try < f_cd:
                    x_cd = x_try
                    f_cd = f_try
                    improved = True
                    # Increase step size for this axis
                    axis_sigma[i] *= 1.3
                    if f_cd < best_f:
                        best_f, best_x = f_cd, x_cd.copy()
                else:
                    # Try negative step
                    x_try = x_cd.copy()
                    x_try[i] -= axis_sigma[i]
                    f_try = safe_call(x_try)
                    calls_in_phase2 += 1

                    if f_try < f_cd:
                        x_cd = x_try
                        f_cd = f_try
                        improved = True
                        # Increase step size for this axis
                        axis_sigma[i] *= 1.3
                        if f_cd < best_f:
                            best_f, best_x = f_cd, x_cd.copy()
                    else:
                        # No improvement in either direction; shrink step
                        axis_sigma[i] *= 0.6

            # If no improvement in a full sweep, shrink all steps
            if not improved:
                axis_sigma *= 0.5
                # If steps are too small, break
                if np.max(axis_sigma) < 1e-8:
                    break

        # Phase 3: High-precision polish
        polish_budget = budget - calls_made
        if polish_budget > 0 and time.time() < deadline:
            x_polish = best_x.copy()
            f_polish = best_f
            polish_sigma = np.full(n, 0.01)
            calls_in_phase3 = 0

            while calls_in_phase3 < polish_budget and time.time() < deadline:
                improved = False
                for i in range(n):
                    if calls_in_phase3 >= polish_budget or time.time() > deadline:
                        break
                    # Try positive step
                    x_try = x_polish.copy()
                    x_try[i] += polish_sigma[i]
                    f_try = safe_call(x_try)
                    calls_in_phase3 += 1

                    if f_try < f_polish:
                        x_polish = x_try
                        f_polish = f_try
                        improved = True
                        polish_sigma[i] *= 1.1
                        if f_polish < best_f:
                            best_f, best_x = f_polish, x_polish.copy()
                    else:
                        # Try negative step
                        x_try = x_polish.copy()
                        x_try[i] -= polish_sigma[i]
                        f_try = safe_call(x_try)
                        calls_in_phase3 += 1

                        if f_try < f_polish:
                            x_polish = x_try
                            f_polish = f_try
                            improved = True
                            polish_sigma[i] *= 1.1
                            if f_polish < best_f:
                                best_f, best_x = f_polish, x_polish.copy()
                        else:
                            polish_sigma[i] *= 0.7

                if not improved:
                    polish_sigma *= 0.5
                    if np.max(polish_sigma) < 1e-10:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
