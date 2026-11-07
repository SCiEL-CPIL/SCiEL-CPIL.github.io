"""Combined Nelder-Mead with adaptive restarts and local polish for BBOB f13."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with adaptive restarts and local polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_fx = float('inf')
        calls = 0

        def time_ok():
            return time.time() < deadline

        def evaluate(x):
            nonlocal calls, best_x, best_fx
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial random sampling to find a good starting region
        # This helps escape poor local minima and provides diverse starting points
        n_init = min(15, budget // 10)
        rng = np.random.default_rng(42)

        for _ in range(n_init):
            if not time_ok():
                break
            x = rng.uniform(-4, 4, dim)
            evaluate(x)

        # Phase 2: Nelder-Mead with adaptive restarts
        # Use multiple restarts if the simplex stagnates
        max_restarts = 5
        restart_count = 0
        prev_best = best_fx

        for restart in range(max_restarts):
            if not time_ok() or calls >= budget:
                break

            # Initialize simplex around current best
            x_center = best_x.copy()
            step = 0.5 if restart == 0 else 0.3 * (0.7 ** restart)

            simplex = [x_center.copy()]
            fx_list = [best_fx]

            for i in range(dim):
                if not time_ok() or calls >= budget:
                    break
                xi = x_center.copy()
                xi[i] += step
                simplex.append(xi)
                fx_list.append(evaluate(xi))

            # One extra point in diagonal direction
            if time_ok() and calls < budget:
                xd = x_center.copy()
                xd += 0.3 * step * np.ones(dim)
                simplex.append(xd)
                fx_list.append(evaluate(xd))

            n_simplex = len(simplex)
            if n_simplex < 2:
                break

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            stall_count = 0
            max_stall = 30

            while calls < budget and time_ok():
                # Sort simplex by function value
                order = np.argsort(fx_list)
                simplex = [simplex[i] for i in order]
                fx_list = [fx_list[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_fx = fx_list[-1]
                best_fx_simplex = fx_list[0]

                # Reflection
                xr = centroid + alpha * (centroid - worst)
                fr = evaluate(xr)

                if fr < best_fx_simplex:
                    # Try expansion
                    if time_ok() and calls < budget:
                        xe = centroid + gamma * (centroid - worst)
                        fe = evaluate(xe)
                        if fe < fr:
                            simplex[-1] = xe
                            fx_list[-1] = fe
                        else:
                            simplex[-1] = xr
                            fx_list[-1] = fr
                    else:
                        simplex[-1] = xr
                        fx_list[-1] = fr
                elif fr < worst_fx:
                    simplex[-1] = xr
                    fx_list[-1] = fr
                else:
                    # Contraction
                    if fr < worst_fx:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (worst - centroid)

                    fc = evaluate(xc)
                    if fc < worst_fx:
                        simplex[-1] = xc
                        fx_list[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n_simplex):
                            if not time_ok() or calls >= budget:
                                break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fx_list[i] = evaluate(simplex[i])

                # Check for convergence
                spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
                if np.all(spread < 1e-6):
                    break

                # Track stagnation
                if best_fx < prev_best - 1e-4:
                    stall_count = 0
                    prev_best = best_fx
                else:
                    stall_count += 1

                if stall_count > max_stall:
                    break

            # If stalled, restart from a new random point
            if stall_count > max_stall and restart < max_restarts - 1:
                if not time_ok() or calls >= budget:
                    break
                x_new = rng.uniform(-4, 4, dim)
                fx_new = evaluate(x_new)
                if fx_new < best_fx:
                    best_x = x_new.copy()
                    best_fx = fx_new
                restart_count += 1

        # Phase 3: Local polish with very small steps
        remaining = budget - calls
        if remaining > 0 and time_ok():
            local_sigma = 0.01
            for _ in range(min(remaining, 50)):
                if not time_ok():
                    break
                y = np.clip(best_x + local_sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                if fy < best_fx:
                    local_sigma *= 1.5
                else:
                    local_sigma *= 0.7
                local_sigma = max(local_sigma, 0.001)

        return best_x, best_fx
# EVOLVE-BLOCK-END
