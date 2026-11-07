"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Nelder-Mead + CMA-ES + coordinate descent for Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # Track calls and best
        calls_made = 0
        best_x = None
        best_f = np.inf

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            calls_made += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_left():
            return time.time() < deadline

        # === Phase 1: Nelder-Mead from multiple random starts ===
        # Schwefel is rugged; NM's geometric simplex search handles this well
        nm_budget = min(budget, 200)
        nm_starts = 5
        per_start_budget = nm_budget // nm_starts

        for s in range(nm_starts):
            if not time_left() or calls_made >= budget:
                break

            # Initialize simplex
            center = rng.uniform(-4, 4, dim)
            simplex = [center.copy()]
            for i in range(dim):
                pert = center.copy()
                pert[i] += 0.5 * (1 if i % 2 == 0 else -1)
                simplex.append(pert)

            # Evaluate initial simplex
            simple_f = []
            for v in simplex:
                if not time_left() or calls_made >= budget:
                    break
                simple_f.append(call_func(v))

            if len(simple_f) < dim + 1:
                continue

            # Nelder-Mead iterations
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            max_iter = per_start_budget - (dim + 1)

            for _ in range(max_iter):
                if not time_left() or calls_made >= budget:
                    break

                # Sort by function value
                order = np.argsort(simple_f)
                simplex = [simplex[i] for i in order]
                simple_f = [simple_f[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = simple_f[-1]

                # Reflection
                xr = centroid + alpha * (centroid - worst)
                fr = call_func(xr)

                if fr < simple_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - worst)
                    fe = call_func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        simple_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simple_f[-1] = fr
                elif fr < simple_f[-2]:
                    simplex[-1] = xr
                    simple_f[-1] = fr
                else:
                    # Contraction
                    if fr < worst_f:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        fc = call_func(xc)
                        if fc <= fr:
                            simplex[-1] = xc
                            simple_f[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simple_f[i] = call_func(simplex[i])
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - worst)
                        fc = call_func(xc)
                        if fc <= worst_f:
                            simplex[-1] = xc
                            simple_f[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simple_f[i] = call_func(simplex[i])

        # === Phase 2: CMA-ES refinement from best point found ===
        remaining = budget - calls_made
        if remaining > 20 and time_left():
            cma_budget = min(remaining, 150)
            pop_size = max(4, dim + 1 + int(3 * np.log(dim + 1)))
            sigma = 1.0
            mean = best_x.copy() if best_x is not None else np.zeros(dim)
            C = np.eye(dim)
            step_size = 1.0

            # CMA-ES parameters
            cmu = 1 / (dim + 4)
            cs = (dim + 1.7) / (dim + 9 + cmu)
            ds = 1 + 2 * max(0, np.sqrt((cmu - 1) / (dim + 1)) - 1) + cs
            cc = (4 - dim) / (dim + 4)
            chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            ps = np.zeros(dim)
            pc = np.zeros(dim)
            evals = 0

            while evals < cma_budget and time_left() and calls_made < budget:
                # Generate population
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                samples = []
                for _ in range(pop_size):
                    z = rng.standard_normal(dim)
                    x = mean + sigma * (L @ z)
                    samples.append(x)

                # Evaluate
                fvals = []
                for x in samples:
                    if calls_made >= budget or not time_left():
                        break
                    fvals.append(call_func(x))

                if len(fvals) < pop_size:
                    break

                # Sort
                order = np.argsort(fvals)
                samples = [samples[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Update mean
                weights = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, pop_size + 1))
                weights /= np.sum(weights)
                mueff = np.sum(weights) ** 2 / np.sum(weights ** 2)
                new_mean = np.sum(np.array(weights[:int(mueff)]).reshape(-1, 1) * np.array(samples[:int(mueff)]), axis=0)

                # Update covariance
                diff = np.array(samples[:int(mueff)]) - mean
                C = (1 - cmu) * C + cmu / mueff * (diff.T @ diff)

                # Update step size (1/5 rule simplified)
                success = fvals[0] < fvals[-1]
                if success:
                    sigma *= 1.2
                else:
                    sigma *= 0.85
                sigma = max(sigma, 1e-4)
                sigma = min(sigma, 10.0)

                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (L.T @ ((new_mean - mean) / sigma))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (evals + 1) / pop_size)) < (1.4 + 2 / (dim + 1)) * chi:
                    pc = (1 - cs) * pc + cs * (new_mean - mean) / sigma
                    C = (1 - cmu) * C + cmu / mueff * (diff.T @ diff) + \
                        cmu * (1 - cmu) * (1 + (1 - ds) * cs / (1 - cs)) * (pc[:, None] @ pc[None, :])

                mean = new_mean
                evals += pop_size

        # === Phase 3: Coordinate descent polish ===
        remaining = budget - calls_made
        if remaining > 5 and time_left() and best_x is not None:
            cd_budget = min(remaining, 50)
            x = best_x.copy()
            fx = best_f
            step = 0.1
            iters = cd_budget

            for _ in range(iters):
                if calls_made >= budget or not time_left():
                    break
                improved = False
                for d in range(dim):
                    if calls_made >= budget or not time_left():
                        break
                    # Try positive and negative steps
                    for direction in [1, -1]:
                        if calls_made >= budget or not time_left():
                            break
                        x_new = x.copy()
                        x_new[d] += direction * step
                        x_new = np.clip(x_new, -5, 5)
                        f_new = call_func(x_new)
                        if f_new < fx:
                            x = x_new
                            fx = f_new
                            improved = True
                if not improved:
                    step *= 0.7
                    if step < 1e-4:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
