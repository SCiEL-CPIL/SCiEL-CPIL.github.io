"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with adaptive restarts for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0

        calls_made = [0]

        def safe_call(x):
            if calls_made[0] >= budget:
                return float('inf')
            x = np.clip(x, -5.0, 5.0)
            try:
                val = float(func(x))
                calls_made[0] += 1
                return val
            except:
                calls_made[0] += 1
                return float('inf')

        rng = np.random.default_rng(42)
        best_x = np.zeros(dim)
        best_f = float('inf')

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        max_restarts = 5
        for restart in range(max_restarts):
            if time.time() > deadline or calls_made[0] >= budget:
                break

            # Initialize center point
            if restart == 0:
                x0 = np.zeros(dim)
            else:
                if best_f < float('inf'):
                    x0 = np.clip(best_x + rng.normal(0, 1.5, dim), -5, 5)
                else:
                    x0 = rng.uniform(-4, 4, dim)

            # Build initial simplex with adaptive size based on restart
            init_step = max(0.5, 2.0 / (restart + 1))
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += init_step
                simplex.append(xi)
            simplex = [np.clip(x, -5, 5) for x in simplex]

            # Evaluate initial simplex
            fvals = [safe_call(x) for x in simplex]
            for x, f in zip(simplex, fvals):
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

            # Track simplex size for adaptive scaling
            simplex_size = init_step
            stall_count = 0
            max_stall = 30

            # Nelder-Mead iterations
            while calls_made[0] < budget and time.time() < deadline:
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Update global best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()
                    stall_count = 0
                else:
                    stall_count += 1

                if stall_count > max_stall:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = safe_call(x_ref)

                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = safe_call(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = safe_call(x_con)
                    if f_con < fvals[-1]:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = safe_call(simplex[i])
                        simplex_size *= 0.5

                        # If simplex is too small, break and restart
                        spread = np.max(np.array(simplex), axis=0) - np.min(np.array(simplex), axis=0)
                        if np.max(spread) < 1e-4:
                            break

            # If we broke out due to small simplex, continue to next restart

        return best_x, best_f
# EVOLVE-BLOCK-END
