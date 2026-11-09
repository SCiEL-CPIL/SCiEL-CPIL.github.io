"""Combined program: Nelder-Mead with Simulated Annealing Restarts for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead Simplex with Simulated Annealing Restarts for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 290.0  # Safety margin vs 330s kill

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')
        n_calls = 0

        def timed_out():
            return time.time() > deadline

        def call_f(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget or timed_out():
                return None
            y = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f_val = func(y)
            n_calls += 1
            if f_val < best_f:
                best_f = float(f_val)
                best_x = y.copy()
            return f_val

        def nelder_mead(x0, max_calls, init_step=0.1):
            """
            Nelder-Mead implementation with call budget tracking.
            Returns early if budget or time is exhausted.
            """
            nonlocal n_calls, best_x, best_f

            # Initialize simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0.copy()
            for i in range(1, dim + 1):
                simplex[i] = x0.copy()
                simplex[i][i - 1] += init_step

            # Evaluate initial simplex
            fvals = np.zeros(dim + 1)
            for i in range(dim + 1):
                f = call_f(simplex[i])
                if f is None:
                    return
                fvals[i] = f

            # Sort simplex by fvals
            def sort_simplex():
                nonlocal simplex, fvals
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]

            sort_simplex()

            calls_used = dim + 1
            calls_limit = max_calls

            for _ in range(max_calls):
                if n_calls >= budget or timed_out():
                    break

                # Check convergence
                if np.std(fvals) < 1e-8:
                    break

                # Calculate centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = call_f(x_r)
                if f_r is None:
                    break
                calls_used += 1

                # Sort to see where x_r fits
                if fvals[0] <= f_r < fvals[-1]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                    sort_simplex()
                elif f_r < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = call_f(x_e)
                    if f_e is None:
                        break
                    calls_used += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                    sort_simplex()
                else:
                    # Contraction
                    rho = 0.5
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = call_f(x_c)
                    if f_c is None:
                        break
                    calls_used += 1
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                        sort_simplex()
                    else:
                        # Shrink
                        sigma = 0.5
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f = call_f(simplex[i])
                            if f is None:
                                return
                            fvals[i] = f
                            calls_used += 1
                        sort_simplex()

                if calls_used >= calls_limit:
                    break

        # Multi-start strategy with simulated annealing-like restarts
        # Initial run from origin
        x_start = np.zeros(dim)
        nelder_mead(x_start, max_calls=100, init_step=0.1)

        # If we still have budget, try more restarts
        # Use simulated annealing: temperature decreases, restarts become more local
        temp = 1.0
        restart_count = 0
        max_restarts = 10

        while n_calls < budget and not timed_out() and restart_count < max_restarts:
            # Decide where to restart
            if restart_count == 0:
                # First restart: near the best point found so far
                x_start = best_x + 0.5 * temp * rng.standard_normal(dim)
            else:
                # Subsequent restarts: mix of local and global exploration
                if rng.random() < 0.3:
                    # Global exploration: random point in the box
                    x_start = rng.uniform(-5.0, 5.0, dim)
                else:
                    # Local exploration: near the best point
                    x_start = best_x + 0.5 * temp * rng.standard_normal(dim)

            x_start = np.clip(x_start, -5.0, 5.0)

            # Allocate remaining budget to this restart
            remaining = budget - n_calls
            if remaining < 10:
                break

            # Decrease temperature (annealing)
            temp *= 0.7

            # Run Nelder-Mead with allocated budget
            max_calls_for_run = min(remaining, 80)
            nelder_mead(x_start, max_calls=max_calls_for_run, init_step=0.05 * temp + 0.01)

            restart_count += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
