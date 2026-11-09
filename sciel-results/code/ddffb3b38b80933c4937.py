"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with Nelder-Mead + coordinate descent for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def try_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: (1+1)-ES with adaptive step size
        def es_11(x0, sigma0, max_calls, deadline):
            nonlocal calls, best_x, best_f
            
            x = x0.copy()
            sigma = sigma0
            fx = try_call(x)
            if fx is None:
                return best_x, best_f
            
            best_f_local = fx
            best_x_local = x.copy()
            
            # (1+1)-ES parameters
            c_sigma = 1.0 / (dim + 4)
            c_step = 1.0 / (dim + 1)
            d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((np.sqrt(dim) - 1) / (dim + 1)) - 1.0) + c_sigma
            success_rate_target = 1.0 / (dim + 1)
            
            # Track successes for sigma adaptation
            successes = []
            window = 10
            
            stall_count = 0
            best_f_so_far = fx
            
            while calls < max_calls and time.time() < deadline:
                # Generate offspring
                z = rng.standard_normal(dim)
                y = x + sigma * z
                fy = try_call(y)
                if fy is None:
                    break
                
                if fy < fx:
                    # Success
                    x = y.copy()
                    fx = fy
                    successes.append(1)
                    stall_count = 0
                    if fx < best_f_so_far:
                        best_f_so_far = fx
                        best_x_local = x.copy()
                else:
                    # Failure
                    successes.append(0)
                    stall_count += 1
                
                # Keep only last 'window' successes
                if len(successes) > window:
                    successes = successes[-window:]
                
                # Adapt sigma based on success rate
                if len(successes) >= window:
                    success_rate = np.mean(successes)
                    # Adapt sigma: increase if too many successes, decrease if too few
                    sigma *= np.exp(c_sigma / d_sigma * (success_rate - success_rate_target))
                    sigma = max(sigma, 1e-8)
                    sigma = min(sigma, 10.0)
                
                # Check for stall and restart
                if stall_count > 30 and sigma > 1e-4:
                    # Reset sigma and try from current best
                    sigma = max(sigma * 0.5, 0.1)
                    x = best_x_local.copy()
                    fx = best_f_local
                    stall_count = 0
                    successes = []
                
                if sigma < 1e-7:
                    break
            
            if fx < best_f_local:
                best_f_local = fx
                best_x_local = x.copy()
            
            return best_x_local, best_f_local

        # Phase 2: Nelder-Mead
        def nelder_mead(x0, f0, max_calls, deadline):
            nonlocal calls, best_x, best_f

            simplex = [x0.copy()]
            simplex_f = [f0]

            # Initialize simplex with adaptive step based on local scale
            step = 0.2
            for i in range(dim):
                perturbation = x0.copy()
                perturbation[i] += step
                simplex.append(perturbation)

            for i in range(1, dim + 1):
                fx_i = try_call(simplex[i])
                if fx_i is None:
                    break
                simplex_f.append(fx_i)
                if fx_i < best_f:
                    best_f = fx_i
                    best_x = simplex[i].copy()

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5

            max_iterations = 100
            for _ in range(max_iterations):
                if calls >= max_calls or time.time() >= deadline:
                    break

                sorted_idx = np.argsort(simplex_f)
                simplex = [simplex[i].copy() for i in sorted_idx]
                simplex_f = [simplex_f[i] for i in sorted_idx]

                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()

                spread = max(simplex_f) - min(simplex_f)
                if spread < 1e-12:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = try_call(x_r)
                if f_r is None:
                    break

                if f_r < simplex_f[0]:
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = try_call(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    f_c = try_call(x_c)
                    if f_c is None:
                        break
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex_f[i] = try_call(simplex[i])
                            if simplex_f[i] is None:
                                break
                        if simplex_f[-1] is None:
                            break

            return best_x, best_f

        # Phase 3: Coordinate descent
        def coordinate_descent(x0, max_calls, deadline):
            nonlocal calls, best_x, best_f
            x = x0.copy()
            f = best_f
            step = 0.05
            no_improve = 0

            for _ in range(100):
                if calls >= max_calls or time.time() >= deadline:
                    break
                if no_improve > 30:
                    break

                improved = False
                for i in range(dim):
                    for direction in [1, -1]:
                        x_try = x.copy()
                        x_try[i] += direction * step
                        x_try = np.clip(x_try, -5.0, 5.0)
                        f_try = try_call(x_try)
                        if f_try is None:
                            return best_x, best_f
                        if f_try < f - 1e-12:
                            x = x_try
                            f = f_try
                            best_f = f
                            best_x = x.copy()
                            improved = True
                            no_improve = 0
                            break
                    if calls >= max_calls or time.time() >= deadline:
                        break

                if not improved:
                    step *= 0.5
                    no_improve += 1
                    if step < 1e-8:
                        break
                else:
                    step = max(step * 1.1, 0.005)

            return best_x, best_f

        # Run phases
        # Phase 1: (1+1)-ES with 65% of budget
        es_budget = int(0.65 * budget)
        best_x, best_f = es_11(np.zeros(dim), 2.5, es_budget, deadline)

        # Phase 2: Nelder-Mead with 25% of budget
        if best_x is not None and calls < budget and time.time() < deadline:
            nm_budget = int(0.25 * budget)
            best_x, best_f = nelder_mead(best_x, best_f, nm_budget, deadline)

        # Phase 3: Coordinate descent with remaining budget
        if best_x is not None and calls < budget and time.time() < deadline:
            cd_budget = budget
            best_x, best_f = coordinate_descent(best_x, best_f, cd_budget, deadline)

        return best_x, best_f
# EVOLVE-BLOCK-END
