"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(mu+lambda)-ES with adaptive sigma, restarts, and final polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        # Global best
        global_best_x = None
        global_best_f = np.inf

        # We'll do multiple restarts of a (mu+lambda)-ES
        # With 500 budget, we can afford several restarts
        # Each restart uses a portion of the budget

        # Strategy: 
        # - Use a (4+4)-ES or similar with adaptive step size
        # - Restart when no improvement for a while
        # - Final phase: local refinement around best point

        # Let's use a simpler but effective approach:
        # 1. Several random restarts with (1+1)-ES with adaptive sigma
        # 2. Each restart gets a chunk of budget
        # 3. Track global best

        # Actually, let's do a proper (mu+lambda) ES with restarts
        mu = 4
        lam = 8
        sigma = 2.0
        c_sigma = 1.0 / (5 * (dim + 3))
        c_c = (4 - mu_eff + 1) / (dim + 4 + 2 * mu_eff)
        mu_eff = mu / 2.0
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # For simplicity and robustness, let's use a cleaner approach:
        # Multiple independent (1+1)-ES runs with different starting points
        # and adaptive step sizes, tracking global best

        # Phase 1: Global exploration with multiple restarts
        # Phase 2: Local refinement

        # Let's allocate budget:
        # - 70% for global search with restarts
        # - 30% for local refinement

        global_budget = int(budget * 0.7)
        local_budget = budget - global_budget

        # Global search: multiple (1+1)-ES runs
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Number of restarts based on budget
        n_restarts = max(1, global_budget // 50)
        per_restart = global_budget // n_restarts

        for r in range(n_restarts):
            if calls_made >= global_budget:
                break
            if time.time() > deadline - 30:  # Leave time for local phase
                break

            # Start from a random point, or from current best with perturbation
            if r == 0:
                x = np.random.uniform(-5, 5, dim)
            else:
                # Perturb the best found so far
                if best_x is not None:
                    x = np.clip(best_x + np.random.uniform(-1, 1, dim), -5, 5)
                else:
                    x = np.random.uniform(-5, 5, dim)

            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0
            no_improve = 0
            max_no_improve = 20

            for step in range(per_restart - 1):
                if calls_made >= global_budget:
                    break
                if time.time() > deadline - 30:
                    break

                y = np.clip(x + sigma * np.random.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    no_improve += 1

                if no_improve > max_no_improve:
                    break

        # Phase 2: Local refinement using Nelder-Mead or similar
        if best_x is not None and calls_made < budget:
            # Use a simple gradient-free local search
            # Nelder-Mead simplex method
            x = best_x.copy()
            fx = best_f

            # Initialize simplex
            simplex = [x.copy()]
            f_simplex = [fx]

            # Expand simplex
            for i in range(dim):
                xi = x.copy()
                step = 0.1 if x[i] >= 0 else -0.1
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                calls_made += 1
                simplex.append(xi)
                f_simplex.append(fi)

            # Nelder-Mead iterations
            n_iter = 0
            max_iter = local_budget - 1
            while calls_made < budget and n_iter < max_iter:
                if time.time() > deadline - 5:
                    break

                # Sort simplex
                idx = np.argsort(f_simplex)
                simplex = [simplex[i] for i in idx]
                f_simplex = [f_simplex[i] for i in idx]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_made += 1
                n_iter += 1

                if fr < f_simplex[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls_made += 1
                    n_iter += 1
                    if fe < fr:
                        simplex[-1] = xe
                        f_simplex[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_simplex[-1] = fr
                elif fr < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                else:
                    # Contraction
                    rho = 0.5
                    if fr < f_simplex[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_made += 1
                    n_iter += 1
                    if fc < f_simplex[-1]:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])
                            calls_made += 1
                            n_iter += 1
                            if calls_made >= budget:
                                break
                        if calls_made >= budget:
                            break

                # Update best
                min_f = min(f_simplex)
                min_idx = np.argmin(f_simplex)
                if min_f < best_f:
                    best_f = min_f
                    best_x = simplex[min_idx].copy()

        # Return the best point found
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
