"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and structured initial sampling for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit

        d = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)
        best_x = np.zeros(d)
        best_fx = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made, best_fx, best_x
            if calls_made >= budget:
                return best_fx
            fx = func(x)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = np.array(x, dtype=float).copy()
            return fx

        # Phase 1: Sobol-like uniform sampling (using random for simplicity, low count)
        # We use a small number of initial points to guide the first CMA-ES start
        n_init = 10
        init_points = []
        for _ in range(n_init):
            x = rng.uniform(-4, 4, d)
            fx = call_func(x)
            init_points.append((fx, x.copy()))
        
        if init_points:
            init_points.sort(key=lambda t: t[0])
            x0 = init_points[0][1]
        else:
            x0 = np.zeros(d)

        # Phase 2: CMA-ES
        # Allocate 80% of remaining budget to CMA-ES
        remaining = budget - calls_made
        cma_budget = int(remaining * 0.8)
        
        if cma_budget > 20:
            # CMA-ES parameters
            lam = 9
            mu = lam // 2
            w = np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = 1.0 / np.sum(w ** 2)

            c_sigma = (mueff + 2) / (d + mueff + 5)
            c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
            c_1 = 2 / ((d + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + c_sigma
            chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

            x = np.array(x0, dtype=float).copy()
            sigma = 1.0
            ps = np.zeros(d)
            pc = np.zeros(d)
            C = np.eye(d)
            local_rng = np.random.default_rng(42)

            def invsqrtC_update(C_mat):
                eigvals, eigvecs = np.linalg.eigh(C_mat)
                eigvals = np.clip(eigvals, 1e-20, None)
                return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            invsqrtC = invsqrtC_update(C)

            # Initial population
            pop = np.tile(x, (lam, 1)) + sigma * local_rng.standard_normal((lam, d))
            pop = np.clip(pop, -5, 5)
            fvals = np.array([call_func(xi) for xi in pop])
            
            if calls_made >= budget:
                return best_x, best_fx

            while calls_made + lam <= budget and time.time() < deadline:
                # Sort and select
                order = np.argsort(fvals)
                selected = pop[order[:mu]]

                # Update mean
                x_new = np.sum(selected * w[:, None], axis=0)

                # Update sigma path
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (invsqrtC @ (x_new - x))
                norm_ps = np.linalg.norm(ps)
                hsig = 1 if norm_ps / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (d + 1)) * chi_n else 0

                # Update C path
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (x_new - x)
                norm_pc = np.linalg.norm(pc)
                hsig_c = 1 if norm_pc / np.sqrt(1 - (1 - c_c) ** 2) < (1.4 + 2 / (d + 1)) * chi_n else 0

                # Update covariance
                delta = hsig_c * c_1 * (1 - c_1) * np.outer(pc, pc)
                rank_mu = np.sum([w[i] * np.outer(selected[i] - x, selected[i] - x) for i in range(mu)], axis=0)
                C = (1 - c_1 - c_mu) * C + delta + c_mu * rank_mu
                C = 0.5 * (C + C.T)

                x = x_new

                # Update sigma
                sigma *= np.exp(c_sigma / damps * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 50.0)

                # Update inverse sqrt
                invsqrtC = invsqrtC_update(C)

                # Generate new population
                pop = np.tile(x, (lam, 1)) + sigma * local_rng.standard_normal((lam, d))
                pop = np.clip(pop, -5, 5)
                fvals = np.array([call_func(xi) for xi in pop])

        # Phase 3: Nelder-Mead Polish
        remaining_ls = budget - calls_made
        if remaining_ls > 10 and time.time() < deadline:
            x = best_x.copy()
            step = 0.5
            simplex = [x.copy()]
            for i in range(d):
                xi = x.copy()
                xi[i] += step
                simplex.append(xi)
            
            # Ensure we have enough budget for initial simplex evals
            if calls_made + d <= budget:
                fvals = np.array([call_func(s) for s in simplex])
                
                max_iter = min(remaining_ls // 2, 50)
                for _ in range(max_iter):
                    if calls_made >= budget - 1 or time.time() > deadline:
                        break

                    idx = np.argsort(fvals)
                    simplex = [simplex[i] for i in idx]
                    fvals = fvals[idx]

                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    alpha = 1.0
                    xr = centroid + alpha * (centroid - simplex[-1])
                    xr = np.clip(xr, -5, 5)
                    fxr = call_func(xr)

                    if fxr < fvals[0]:
                        # Expansion
                        gamma = 2.0
                        xe = centroid + gamma * (centroid - simplex[-1])
                        xe = np.clip(xe, -5, 5)
                        fxe = call_func(xe)
                        if fxe < fxr:
                            simplex[-1] = xe
                            fvals[-1] = fxe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fxr
                    elif fxr < fvals[-2]:
                        simplex[-1] = xr
                        fvals[-1] = fxr
                    else:
                        # Contraction
                        rho = 0.5
                        xc = centroid + rho * (simplex[-1] - centroid)
                        xc = np.clip(xc, -5, 5)
                        fxc = call_func(xc)
                        if fxc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fxc
                        else:
                            # Shrink
                            for i in range(1, d + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                if calls_made < budget:
                                    fvals[i] = call_func(simplex[i])

        return best_x, best_fx
# EVOLVE-BLOCK-END
