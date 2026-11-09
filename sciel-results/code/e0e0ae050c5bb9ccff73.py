import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls = [0]
        best_val = [np.inf]
        best_x = [np.zeros(n)]

        def safe_call(x):
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if calls[0] >= budget:
                raise MemoryError("Budget exhausted")
            calls[0] += 1
            val = func(x)
            if val < best_val[0]:
                best_val[0] = val
                best_x[0] = x.copy()
            return val

        def nelder_mead(x0, max_iter=100, tol=1e-10):
            """
            Simple Nelder-Mead implementation with call counting and time checking.
            """
            if time.time() > deadline or calls[0] >= budget:
                return x0, best_val[0]

            # Initialize simplex
            x = np.array(x0, dtype=float)
            f0 = safe_call(x)
            simplex = [x.copy()]
            fvals = [f0]

            # Create initial simplex
            for i in range(n):
                xi = x.copy()
                if xi[i] != 0:
                    xi[i] += 0.05 * abs(xi[i])
                else:
                    xi[i] += 0.05
                fi = safe_call(xi)
                simplex.append(xi)
                fvals.append(fi)

            # Sort
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for it in range(max_iter):
                if time.time() > deadline or calls[0] >= budget:
                    break

                # Check convergence
                if abs(fvals[0] - fvals[-1]) < tol:
                    break

                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = safe_call(xr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = safe_call(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (xr - centroid)
                    fc = safe_call(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = safe_call(simplex[i])

                # Sort again
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

            return simplex[0], fvals[0]

        # Phase 1: Global Search with Simulated Annealing
        # We split the budget: ~70% for SA, ~30% for Polish
        sa_budget = int(budget * 0.7)
        polish_budget = budget - sa_budget

        n_restarts = 5
        budget_per_restart = sa_budget // n_restarts

        for r in range(n_restarts):
            if time.time() > deadline or calls[0] >= budget:
                break

            # Initialize random point
            x = rng.uniform(-5, 5, n)
            fx = safe_call(x)

            # SA parameters
            T = 5.0
            alpha = 0.97
            sigma = 1.5

            for i in range(budget_per_restart):
                if time.time() > deadline or calls[0] >= budget:
                    break

                # 10% chance of a larger jump
                if rng.random() < 0.1:
                    step_mult = 2.0
                else:
                    step_mult = 1.0

                y = x + sigma * step_mult * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = safe_call(y)

                delta = fy - fx

                # Acceptance probability
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    x = y
                    fx = fy
                    # Adapt step size: increase on success
                    sigma *= 1.02
                else:
                    # Adapt step size: decrease on failure
                    sigma *= 0.98

                T *= alpha

        # Phase 2: Local Polish around the best point found
        if time.time() < deadline and calls[0] < budget:
            x_polish = best_x[0].copy()
            # Run Nelder-Mead for polish
            try:
                x_best, f_best = nelder_mead(x_polish, max_iter=50, tol=1e-10)
            except Exception:
                pass

        return best_x[0], best_val[0]
