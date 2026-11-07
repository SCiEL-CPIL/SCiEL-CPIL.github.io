"""CMA-ES with multi-start and coordinate-descent polish for BBOB-style f6."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-start restarts and coordinate-descent local search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # 15s safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return calls_made < budget - 1

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def cma_es(x0, f0, max_iters, sigma0, population_size):
            """Run CMA-ES starting from x0 with initial sigma."""
            nonlocal calls_made

            n = dim
            m = population_size

            # CMA-ES parameters
            mu = m // 2  # number of weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Learning rates
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            # Chi-squared approximation
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Initialize
            mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)

            # Initial population
            B, D = np.linalg.eigh(C)
            D_inv = 1.0 / np.sqrt(D)

            for iteration in range(max_iters):
                if not time_ok() or not budget_ok():
                    break

                # Generate population
                pop = []
                fvals = []
                for i in range(m):
                    z = np.random.standard_normal(n)
                    y = (B * D_inv) @ z
                    x_i = np.clip(mean + sigma * y, -5, 5)
                    f_i = func(x_i)
                    calls_made += 1
                    pop.append(x_i)
                    fvals.append(f_i)
                    update_best(x_i, f_i)

                # Sort by function value
                idx = np.argsort(fvals)
                pop = [pop[i] for i in idx]
                fvals = [fvals[i] for i in idx]

                # Update mean
                mean_old = mean.copy()
                mean = sum(weights[i] * pop[i] for i in range(mu))

                # Update covariance matrix adaptation
                # ps update
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D_inv * (B.T @ (mean - mean_old))))

                # Check norm of ps
                if np.linalg.norm(ps) / chi_n < (1.4 + 2 / (n + 1)) * np.sqrt(1 - (1 - c_c) ** (iteration + 1)):
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - mean_old) / sigma
                    C = (1 - c_1 - c_mu) * C + c_1 * (2 - c_1) * C + c_mu / sigma ** 2 * (
                        c_1 * (2 - c_1) * np.outer(pc, pc) + (1 - c_1) * (pc @ pc.T - C)
                    )

                # Symmetrize C
                C = (C + C.T) / 2

                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = min(sigma, 1e10)  # cap to prevent explosion

                # Recompute B, D
                B, D = np.linalg.eigh(C)
                D_inv = 1.0 / np.sqrt(np.maximum(D, 1e-16))

                # Convergence check
                if np.linalg.norm(mean - mean_old) < 1e-10 * max(1, sigma):
                    break

                # If sigma is too small, break
                if sigma < 1e-8:
                    break

            return mean, fvals[0] if fvals else np.inf

        def coordinate_descent(x, fx, max_iters, initial_step):
            """Adaptive coordinate descent local search."""
            nonlocal calls_made

            n = dim
            steps = np.full(n, initial_step)
            x = x.copy()

            for iteration in range(max_iters):
                if not time_ok() or not budget_ok():
                    break

                improved = False
                for i in range(n):
                    # Try positive direction
                    x_pos = x.copy()
                    x_pos[i] += steps[i]
                    x_pos = np.clip(x_pos, -5, 5)
                    f_pos = func(x_pos)
                    calls_made += 1

                    if f_pos < fx:
                        x = x_pos
                        fx = f_pos
                        steps[i] *= 1.2
                        improved = True
                        update_best(x, fx)
                    else:
                        # Try negative direction
                        x_neg = x.copy()
                        x_neg[i] -= steps[i]
                        x_neg = np.clip(x_neg, -5, 5)
                        f_neg = func(x_neg)
                        calls_made += 1

                        if f_neg < fx:
                            x = x_neg
                            fx = f_neg
                            steps[i] *= 1.2
                            improved = True
                            update_best(x, fx)
                        else:
                            steps[i] *= 0.7

                    if steps[i] < 1e-8:
                        steps[i] = 1e-8

                if not improved and iteration > 10:
                    # No improvement in a full cycle, shrink all steps
                    steps *= 0.5
                    if np.all(steps < 1e-8):
                        break

            return x, fx

        # Phase 1: Initial random exploration to find good starting points
        n_initial = 8
        initial_budget = min(40, budget // 10)
        rng = np.random.default_rng(42)

        candidates = []
        for i in range(n_initial):
            if not time_ok() or not budget_ok():
                break
            x = rng.uniform(-4, 4, dim)
            f = func(x)
            calls_made += 1
            update_best(x, f)
            candidates.append((f, x))

        # Sort candidates by function value
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Run CMA-ES from top candidates
        population_size = 10
        n_restarts = min(4, len(candidates))

        for i in range(n_restarts):
            if not time_ok() or not budget_ok():
                break

            # Allocate remaining budget to remaining restarts
            remaining = budget - calls_made
            restarts_left = n_restarts - i
            restart_budget = remaining // restarts_left
            if restart_budget < 20:
                restart_budget = 20

            max_iters = max(5, restart_budget // population_size)

            x_start = candidates[i][1]
            f_start = candidates[i][0]

            # Start CMA-ES with adaptive initial sigma
            sigma0 = 1.0
            if i == 0:
                sigma0 = 2.0  # wider search for first restart
            else:
                sigma0 = 0.5  # narrower for subsequent

            mean, f_best = cma_es(x_start, f_start, max_iters, sigma0, population_size)

            # Add result to candidates if better
            if f_best < best_f:
                candidates.insert(0, (f_best, mean))

        # Phase 3: Coordinate descent polish on the best point found
        if best_x is not None and time_ok() and budget_ok():
            x_polished, f_polished = coordinate_descent(
                best_x.copy(), best_f,
                max_iters=50,
                initial_step=0.1
            )

        # Ensure we have a valid result
        if best_x is None:
            x = np.zeros(dim)
            best_x = x
            if budget_ok():
                best_f = func(x)
                calls_made += 1
            else:
                best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
