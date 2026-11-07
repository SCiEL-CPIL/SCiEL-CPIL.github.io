"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid global-local optimizer: coarse grid + Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        def time_ok():
            return time.time() < deadline

        x_best = np.zeros(self.dim)
        f_best = np.inf
        calls = 0
        budget = self.budget

        def evaluate(x):
            nonlocal x_best, f_best, calls
            if calls >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        # Phase 1: Coarse random search to find promising regions
        # Use a stratified sampling approach for better coverage
        rng = np.random.default_rng(42)

        # Start with a few random points
        n_init = min(10, budget)
        for _ in range(n_init):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, self.dim)
            evaluate(x)

        # Phase 2: Nelder-Mead refinement from the best point
        # Nelder-Mead uses dim+1 function evaluations per iteration typically
        # but can be more efficient in practice

        def nelder_mead(x0, f0, max_iter=100):
            nonlocal calls, budget, f_best, x_best
            if calls >= budget or not time_ok():
                return

            n = self.dim
            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [f0]

            # Create initial simplex
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.5 if x0[i] < 4.5 else -0.5
                xi = np.clip(xi, -5.0, 5.0)
                fi = evaluate(xi)
                if fi is None:
                    return
                simplex.append(xi)
                fvals.append(fi)

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for _ in range(max_iter):
                if calls >= budget or not time_ok():
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                x_worst = simplex[-1]
                f_worst = fvals[-1]
                x_best_sm = simplex[0]
                f_best_sm = fvals[0]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = evaluate(x_r)
                if f_r is None:
                    break

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = evaluate(x_e)
                    if f_e is None:
                        break
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
                        x_c = centroid - rho * (x_r - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = evaluate(x_c)
                    if f_c is None:
                        break
                    if f_c <= min(f_r, f_worst):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = x_best_sm + sigma * (simplex[i] - x_best_sm)
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fi = evaluate(simplex[i])
                            if fi is None:
                                return
                            fvals[i] = fi

        # Run Nelder-Mead from the best point found
        nelder_mead(x_best, f_best, max_iter=200)

        # Phase 3: If we have budget left, try a few more restarts from perturbed best
        remaining = budget - calls
        if remaining > 20 and time_ok():
            n_restarts = min(5, remaining // 20)
            for r in range(n_restarts):
                if not time_ok() or calls >= budget:
                    break
                # Perturb the best point
                x_restart = x_best + rng.normal(0, 1.0, self.dim)
                x_restart = np.clip(x_restart, -5.0, 5.0)
                f_restart = evaluate(x_restart)
                if f_restart is None:
                    break
                # Run a short Nelder-Mead from here
                nelder_mead(x_restart, f_restart, max_iter=30)

        return x_best, f_best
# EVOLVE-BLOCK-END
