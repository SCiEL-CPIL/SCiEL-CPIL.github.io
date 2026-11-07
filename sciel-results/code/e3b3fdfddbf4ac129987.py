"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with multi-start and final polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin

        rng = np.random.default_rng(42)

        def time_left():
            return deadline - time.time()

        # Phase 1: Multi-start exploration (use ~20% of budget)
        n_starts = 8
        explore_budget = min(n_starts * 2, budget // 5)
        best_x = None
        best_f = np.inf
        calls = 0

        for s in range(n_starts):
            if calls >= explore_budget or time_left() < 5:
                break
            # Deterministic spread points + random
            if s == 0:
                x0 = np.zeros(dim)
            elif s == 1:
                x0 = np.ones(dim) * 2.0
            elif s == 2:
                x0 = -np.ones(dim) * 2.0
            elif s == 3:
                x0 = rng.uniform(-4, 4, dim)
            else:
                x0 = rng.uniform(-4, 4, dim)
            
            f0 = func(x0)
            calls += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            # Quick local improvement from this start
            for _ in range(2):
                if calls >= explore_budget or time_left() < 5:
                    break
                step = 0.5 * rng.standard_normal(dim)
                y = np.clip(best_x + step, -5, 5)
                fy = func(y)
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

        x = best_x.copy()
        fx = best_f

        # Phase 2: CMA-ES refinement
        # Reserve ~15% of remaining budget for polish
        remaining = budget - calls
        cma_budget = int(remaining * 0.85)
        polish_budget = remaining - cma_budget

        # CMA-ES parameters
        lam = 10
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        c_sigma = (lam + 2) / (dim + lam + 5)
        c_c = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lam)
        c_mu = min(1 - c_1, 2 * ((lam / 2 + 1) / (dim + 2) - 1) / ((dim + 2) ** 2 + lam))
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        sigma = 1.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)

        cma_calls = 0
        while cma_calls < cma_budget and calls < budget:
            if time_left() < 5:
                break

            # Generate population
            offspring = []
            for i in range(lam):
                if calls >= budget or time_left() < 3:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                cma_calls += 1
                offspring.append((y, fy))

            if len(offspring) < mu:
                break

            offspring.sort(key=lambda t: t[1])

            # Update best
            if offspring[0][1] < fx:
                fx = offspring[0][1]
                x = offspring[0][0].copy()

            # Mean of selected
            x_new = sum(weights[i] * offspring[i][0] for i in range(mu))

            # Delta
            diff = (x_new - x) / (sigma + 1e-16)

            # Evolution path for sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff / dim) * diff

            # Evolution path for covariance
            pc = (1 - c_c) * pc + c_c * diff

            # Covariance matrix update
            H = np.zeros((dim, dim))
            for i in range(mu):
                diff_i = (offspring[i][0] - x) / (sigma + 1e-16)
                H += weights[i] * np.outer(diff_i, diff_i)

            C = (1 + c_1 + c_mu) * C - c_1 * np.outer(pc, pc) - c_mu * H

            # Ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            D = np.sqrt(eigvals)
            B = eigvecs

            # Step size adaptation
            sigma *= np.exp(c_sigma / chi * (np.linalg.norm(ps) / chi - 1))
            sigma = max(sigma, 1e-6)

            x = x_new

        # Phase 3: Nelder-Mead polish with remaining budget
        if calls < budget and time_left() > 3:
            # Initialize simplex
            simplex = [x.copy()]
            for d in range(dim):
                pt = x.copy()
                step = 0.1 if abs(x[d]) > 0.5 else 0.1
                pt[d] += step
                pt = np.clip(pt, -5, 5)
                simplex.append(pt)
            
            fvals = [fx]
            for pt in simplex[1:]:
                if calls >= budget or time_left() < 2:
                    break
                fv = func(pt)
                calls += 1
                fvals.append(fv)
                if fv < fx:
                    fx = fv
                    x = pt.copy()

            # Nelder-Mead iterations
            max_nm_iters = 50
            for _ in range(max_nm_iters):
                if calls >= budget or time_left() < 2:
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                if calls >= budget or time_left() < 2:
                    break
                fr = func(xr)
                calls += 1

                if fr < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    if calls >= budget or time_left() < 2:
                        break
                    fe = func(xe)
                    calls += 1
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
                    rho = 0.5
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    if calls >= budget or time_left() < 2:
                        break
                    fc = func(xc)
                    calls += 1
                    if fc < max(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if calls >= budget or time_left() < 2:
                                break
                            fi = func(simplex[i])
                            calls += 1
                            fvals[i] = fi

                # Update best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < fx:
                    fx = fvals[min_idx]
                    x = simplex[min_idx].copy()

        return x, fx
# EVOLVE-BLOCK-END
