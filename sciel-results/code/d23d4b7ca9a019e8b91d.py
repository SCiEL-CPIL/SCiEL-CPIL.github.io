import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls = 0
        history = []

        def try_eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return None
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            history.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def nelder_mead(x0, max_iter, initial_radius=0.5):
            nonlocal calls, best_x, best_f
            if max_iter <= 0:
                return x0, best_f

            # Initialize simplex
            simplex = [x0.copy() for _ in range(n + 1)]
            simplex[0] = x0
            for i in range(1, n + 1):
                simplex[i] = x0.copy()
                simplex[i][i-1] += initial_radius
            
            f_values = [None] * (n + 1)
            for i in range(n + 1):
                f_values[i] = try_eval(simplex[i])
                if f_values[i] is None:
                    return x0, best_f

            # Sort simplex
            order = np.argsort(f_values)
            simplex = [simplex[i] for i in order]
            f_values = [f_values[i] for i in order]

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            for _ in range(max_iter):
                if calls >= budget - 5 or time.time() > deadline - 2:
                    break

                # 1. Reflection
                centroid = np.mean(simplex[:-1], axis=0)
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = try_eval(x_r)
                if f_r is None:
                    break

                if f_r < f_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = try_eval(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    # Reflection is good
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_values[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                    
                    f_c = try_eval(x_c)
                    if f_c is None:
                        break

                    if f_c < min(f_r, f_values[-1]):
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_shrunk = try_eval(simplex[i])
                            if f_shrunk is None:
                                return x0, best_f
                            f_values[i] = f_shrunk

                # Re-sort
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]

                # Check for convergence
                if np.std(f_values) < 1e-6:
                    break

            return simplex[0], best_f

        # Phase 1: Global Sampling
        # Use Latin Hypercube Sampling for better coverage
        num_global = 80
        lhs_points = np.zeros((num_global, n))
        for i in range(n):
            perm = rng.permutation(num_global)
            lhs_points[:, i] = (perm + rng.random(num_global)) / num_global
        lhs_points = lhs_points * 10.0 - 5.0

        for x in lhs_points:
            try_eval(x)
            if time.time() > deadline - 20:
                break

        # Add center
        try_eval(np.zeros(n))

        # Add random points
        for _ in range(10):
            if calls >= 100 or time.time() > deadline - 20:
                break
            try_eval(rng.uniform(-5, 5, n))

        # Phase 2: Local Optimization
        if history:
            # Sort by function value
            history.sort(key=lambda t: t[0])
            starts = []
            for f, x in history:
                is_dup = False
                for s in starts:
                    if np.linalg.norm(x - s) < 1.0:
                        is_dup = True
                        break
                if not is_dup:
                    starts.append(x)
                if len(starts) == 5:
                    break

            remaining_budget = budget - calls
            if remaining_budget > 0 and starts:
                budget_per_start = remaining_budget // len(starts)
                for x_start in starts:
                    if calls >= budget - 5 or time.time() > deadline - 5:
                        break
                    alloc = min(budget_per_start, budget - calls)
                    if alloc < 10:
                        break
                    nelder_mead(x_start, alloc)

        return best_x, best_f
