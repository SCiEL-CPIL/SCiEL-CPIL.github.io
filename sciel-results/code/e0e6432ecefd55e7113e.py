"""Direct optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex optimizer with multi-restart for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def nelder_mead(start_x, max_calls=100, max_time=280.0):
            nonlocal best_x, best_f, calls_made
            
            # Initialize simplex: start point + dim perturbed points
            simplex = [start_x.copy()]
            for i in range(dim):
                v = start_x.copy()
                v[i] += 0.5 * (1.0 if start_x[i] < 4.5 else -1.0)
                v[i] = np.clip(v[i], -5, 5)
                simplex.append(v)
            
            # Evaluate initial simplex
            fvals = []
            for x in simplex:
                if calls_made >= budget or time.time() > max_time:
                    break
                f = func(x)
                calls_made += 1
                fvals.append(f)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            if len(fvals) < dim + 1:
                return

            # Nelder-Mead iterations
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            max_iter = max_calls
            no_improve = 0
            threshold = 1e-8

            for _ in range(max_iter):
                if calls_made >= budget or time.time() > max_time:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check convergence
                if abs(fvals[0] - fvals[-1]) < threshold:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                f_worst = fvals[-1]

                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5, 5)
                if calls_made >= budget or time.time() > max_time:
                    break
                f_r = func(x_r)
                calls_made += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    if calls_made >= budget or time.time() > max_time:
                        break
                    f_e = func(x_e)
                    calls_made += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    if calls_made >= budget or time.time() > max_time:
                        break
                    f_c = func(x_c)
                    calls_made += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                    if f_c < min(f_r, f_worst):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                        for i in range(1, len(simplex)):
                            if calls_made >= budget or time.time() > max_time:
                                break
                            fvals[i] = func(simplex[i])
                            calls_made += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()
                        no_improve += 1
                        if no_improve > 5:
                            break

                # Check for improvement
                if fvals[0] < fvals[-1] * 0.999 or fvals[0] < fvals[-1] - 1e-6:
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > 10:
                        break

            return

        # Multi-start Nelder-Mead
        starts = [
            np.zeros(dim),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
            np.ones(dim) * 3.0,
            np.ones(dim) * -3.0,
            np.array([2.5, -2.5, 2.5, -2.5, 2.5]),
            np.array([-2.5, 2.5, -2.5, 2.5, -2.5]),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
        ]

        # Budget per start: allocate remaining budget / remaining starts
        n_starts = len(starts)
        
        for idx, start in enumerate(starts):
            if calls_made >= budget or time.time() > deadline:
                break
            
            remaining_starts = n_starts - idx
            remaining_budget = budget - calls_made
            # Allocate at least 20 calls per start, cap at 80
            calls_per_start = max(20, min(80, int(remaining_budget / remaining_starts)))
            
            nelder_mead(start, max_calls=calls_per_start, max_time=deadline)
            
            # If we've used most of the budget, stop
            if calls_made > budget - 15:
                break

        # Ensure we have a valid result
        if best_x is None:
            x = np.zeros(dim)
            best_x = x
            best_f = func(x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
