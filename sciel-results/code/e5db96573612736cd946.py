import numpy as np
import time
import scipy.optimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = None

    def __call__(self, func):
        start_time = time.time()
        self.deadline = start_time + 285  # Safety margin out of 300s

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def eval_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            f = float(func(x_clipped))
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Phase 1: Initial sampling with more diverse starts
        n_starts = 15
        candidates = [np.zeros(dim)]
        # Add points near the center and spread out
        for _ in range(n_starts - 1):
            candidates.append(rng.uniform(-4, 4, dim))

        for x0 in candidates:
            if calls >= budget or time.time() > self.deadline:
                break
            x0 = np.clip(x0, -5, 5)
            f0 = eval_func(x0)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = eval_func(best_x)

        # Phase 2: L-BFGS-B from best point for fast convergence
        while calls < budget and time.time() < self.deadline:
            x_start = best_x.copy()

            def scipy_func(x):
                return eval_func(x)

            remaining = budget - calls
            if remaining < 10:
                break

            try:
                res = scipy.optimize.minimize(
                    scipy_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 300, 'maxfun': remaining, 'ftol': 1e-14, 'gtol': 1e-10}
                )
            except Exception:
                break

            # Update best if L-BFGS-B found something better
            if res.fun < best_f:
                best_f = res.fun
                best_x = np.clip(res.x, -5, 5).copy()

            if res.success:
                break

            if budget - calls > 20:
                continue
            else:
                break

        # Phase 3: Multi-scale SLSQP refinement with tighter tolerances
        if calls < budget and time.time() < self.deadline:
            scales = [0.3, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            for scale in scales:
                if calls >= budget or time.time() > self.deadline:
                    break
                remaining = budget - calls
                if remaining < 5:
                    break

                # Generate a few perturbed starts around the best point
                n_perturb = min(4, remaining // 10)
                if n_perturb < 1:
                    n_perturb = 1

                best_local_x = best_x.copy()
                best_local_f = best_f

                for _ in range(n_perturb):
                    if calls >= budget or time.time() > self.deadline:
                        break
                    perturbation = rng.normal(0, scale, dim)
                    x_start = best_x + perturbation
                    x_start = np.clip(x_start, -5, 5)
                    f_start = eval_func(x_start)

                    if f_start < best_local_f:
                        best_local_f = f_start
                        best_local_x = x_start.copy()

                    # Try SLSQP from this point with tight tolerance
                    if calls >= budget or time.time() > self.deadline:
                        break
                    remaining = budget - calls
                    if remaining < 5:
                        break

                    try:
                        res = scipy.optimize.minimize(
                            eval_func,
                            x_start,
                            method='SLSQP',
                            bounds=[(-5, 5)] * dim,
                            options={'maxiter': 100, 'ftol': 1e-12, 'maxfun': min(remaining, 100)}
                        )
                    except Exception:
                        continue

                    # Update best if SLSQP found something better
                    if res.fun < best_local_f:
                        best_local_f = res.fun
                        best_local_x = np.clip(res.x, -5, 5).copy()

                # Update global best from local best
                if best_local_f < best_f:
                    best_f = best_local_f
                    best_x = best_local_x.copy()

        # Phase 4: Final high-precision polish with very tight tolerance
        if calls < budget and time.time() < self.deadline:
            remaining = budget - calls
            if remaining >= 5:
                try:
                    res = scipy.optimize.minimize(
                        eval_func,
                        best_x,
                        method='SLSQP',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 200, 'ftol': 1e-15, 'maxfun': min(remaining, 200)}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5).copy()
                except Exception:
                    pass

        return best_x, best_f
