"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # Global best tracking
        best_x = None
        best_f = float('inf')
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            calls_made += 1
            return float(func(x))

        # CMA-ES implementation
        def cma_es(func_eval, x0, sigma0, max_calls, deadline):
            nonlocal best_x, best_f, calls_made
            dim = self.dim
            n = dim
            # CMA-ES parameters
            lambda_ = max(4, int(4 + 3 * np.log(n)))
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            x = np.array(x0, dtype=np.float64)
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)

            # Initial population
            rng = np.random.default_rng(42)
            pop = []
            pop_f = []
            for i in range(lambda_):
                xi = x + sigma * (B @ (D * rng.standard_normal(n)))
                xi = np.clip(xi, -5.0, 5.0)
                fi = func_eval(xi)
                pop.append(xi)
                pop_f.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            calls_made += lambda_

            # Sort by fitness
            order = np.argsort(pop_f)
            pop = [pop[i] for i in order]
            pop_f = [pop_f[i] for i in order]

            while calls_made < max_calls and time.time() < deadline:
                # Update mean
                m_new = np.sum([weights[i] * pop[i] for i in range(mu)], axis=0)
                # Update path evolution strategies
                pc = (1 - c_c) * pc + c_c * (m_new - x) / (sigma * chi_n)
                cs = np.sqrt((1 + 2 * c_sigma / d_sigma - c_sigma) * (1 - c_c))
                hs = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (calls_made // lambda_))) < (1.4 + 2 / (n + 1)) * chi_n
                ps = (1 - c_sigma) * ps + hs * c_sigma * cs * (m_new - x) / sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 50.0)

                # Update covariance matrix
                delta_hs = (1 - hs) * c_sigma * (2 - c_sigma)
                rank_one = np.outer(ps, ps)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    zi = (pop[i] - x) / sigma
                    rank_mu += weights[i] * np.outer(zi, zi)
                C = (1 + c_1 * delta_hs - c_1 - c_mu) * C + c_1 * (pc @ pc.T + delta_hs * np.eye(n)) + c_mu * rank_mu

                # Ensure positive semi-definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-15)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update B and D
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-15)
                D = np.sqrt(eigvals)
                B = eigvecs
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                x = m_new
                x = np.clip(x, -5.0, 5.0)

                # Generate new population
                new_pop = []
                new_pop_f = []
                for i in range(lambda_):
                    xi = x + sigma * (B @ (D * rng.standard_normal(n)))
                    xi = np.clip(xi, -5.0, 5.0)
                    fi = func_eval(xi)
                    new_pop.append(xi)
                    new_pop_f.append(fi)
                    if fi < best_f:
                        best_f = fi
                        best_x = xi.copy()

                calls_made += lambda_

                # Sort
                order = np.argsort(new_pop_f)
                pop = [new_pop[i] for i in order]
                pop_f = [new_pop_f[i] for i in order]

            return best_x, best_f

        # Nelder-Mead polish
        def nelder_mead(func_eval, x0, max_calls, deadline):
            nonlocal best_x, best_f, calls_made
            dim = self.dim
            n = dim
            # Initialize simplex
            simplex = [np.array(x0, dtype=np.float64)]
            for i in range(n):
                point = simplex[0].copy()
                if point[i] != 0:
                    point[i] *= 1.05
                else:
                    point[i] = 0.01
                point = np.clip(point, -5.0, 5.0)
                simplex.append(point)

            fvals = [func_eval(p) for p in simplex]
            calls_made += n + 1
            for f, p in zip(fvals, simplex):
                if f < best_f:
                    best_f = f
                    best_x = p.copy()

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5
            max_iter = max(1, (max_calls - calls_made) // (n + 1))

            for _ in range(max_iter):
                if time.time() >= deadline or calls_made >= max_calls:
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()

                # Centroid of all but worst
                centroid = np.sum(simplex[:n], axis=0) / n

                # Reflection
                xr = centroid + alpha * (centroid - simplex[n])
                xr = np.clip(xr, -5.0, 5.0)
                fr = func_eval(xr)
                calls_made += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[n])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = func_eval(xe)
                    calls_made += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[n] = xe
                        fvals[n] = fe
                    else:
                        simplex[n] = xr
                        fvals[n] = fr
                elif fr < fvals[n - 1]:
                    simplex[n] = xr
                    fvals[n] = fr
                else:
                    # Contraction
                    if fr < fvals[n]:
                        # Outside contraction
                        xc = centroid + 0.5 * (simplex[n] - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (centroid - simplex[n])
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = func_eval(xc)
                    calls_made += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()

                    if fc < min(fr, fvals[n]):
                        simplex[n] = xc
                        fvals[n] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = func_eval(simplex[i])
                            calls_made += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()

            return best_x, best_f

        # Multi-start CMA-ES with restarts
        rng = np.random.default_rng(123)
        n_starts = 5
        calls_per_start = budget // n_starts

        for s in range(n_starts):
            if calls_made >= budget or time.time() >= deadline:
                break

            # Initial point: random or perturbed best
            if s == 0:
                x0 = rng.uniform(-5, 5, dim)
                sigma0 = 2.0
            else:
                if best_x is not None:
                    x0 = np.clip(best_x + rng.normal(0, 1.0, dim), -5, 5)
                else:
                    x0 = rng.uniform(-5, 5, dim)
                sigma0 = 1.0

            remaining = budget - calls_made
            if remaining <= 0:
                break

            # Reserve some calls for polish
            polish_budget = max(50, remaining // 5)
            cma_budget = remaining - polish_budget

            if cma_budget < 10:
                break

            cma_es(safe_func, x0, sigma0, calls_made + cma_budget, deadline)

            # Nelder-Mead polish
            if calls_made < budget and time.time() < deadline:
                if best_x is not None:
                    nelder_mead(safe_func, best_x, budget, deadline)

        # Final polish if we have budget left
        if best_x is not None and calls_made < budget and time.time() < deadline:
            nelder_mead(safe_func, best_x, budget, deadline)

        return best_x, best_f
# EVOLVE-BLOCK-END
