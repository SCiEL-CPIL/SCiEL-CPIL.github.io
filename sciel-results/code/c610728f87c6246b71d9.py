"""Hybrid CMA-ES + (1+1)-ES with coordinate-descent polish for Schwefel f20."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Hybrid CMA-ES/(1+1)-ES with coordinate-descent polish on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # safety margin before 330s kill
        calls = 0

        def F(x):
            nonlocal calls
            calls += 1
            return func(x)

        rng = np.random.default_rng(42)

        # Initial point: center with small random offset
        best_x = rng.uniform(-0.5, 0.5, dim)
        best_f = F(best_x)

        def time_ok():
            return time.time() < deadline - 3.0

        # ---- Phase 1: CMA-ES for global search (first ~60% of budget) ----
        def cma_es_phase(x0, f0, budget_alloc):
            nonlocal best_x, best_f, calls
            n = dim
            m = int(4 + 3 * np.log(n))  # population size
            sigma = 1.5  # initial step size
            C = np.eye(n)
            ps = np.zeros(n)
            pc = np.zeros(n)

            mu = m // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / w.sum()
            c_sigma = (mueff + 2) / ((n + 2) * (mueff + 2))
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / ((n + 4) * (4 + mueff / n))
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            ev, V = np.linalg.eigh(C)
            ev = np.maximum(ev, 1e-20)
            D = np.sqrt(ev)
            B = V
            invsqrtC = V @ np.diag(1.0 / np.sqrt(ev)) @ V.T

            x = x0.copy()
            fx = f0
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            stall_count = 0
            last_improvement_f = best_f

            while calls < budget_alloc and time_ok():
                z = rng.standard_normal((m, n))
                off = (z * D) @ B.T
                off *= sigma
                y = x[None, :] + off
                y = np.clip(y, -5, 5)

                fy = np.empty(m)
                for i in range(m):
                    if calls >= budget_alloc or not time_ok():
                        break
                    fy[i] = F(y[i])

                if calls >= budget_alloc:
                    break

                order = np.argsort(fy)
                fy_sorted = fy[order]
                y_sorted = y[order]

                if fy_sorted[0] < best_f:
                    best_f = fy_sorted[0]
                    best_x = y_sorted[0].copy()

                # Check for stall
                if fy_sorted[0] < last_improvement_f - 1e-8:
                    last_improvement_f = fy_sorted[0]
                    stall_count = 0
                else:
                    stall_count += 1

                # Recompute mean of best mu
                mean_new = np.sum(w[:, None] * y_sorted[:mu], axis=0)

                # Evolution path sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (
                    invsqrtC @ (mean_new - x) / sigma
                )
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 1e4)

                # Covariance update
                yw = y_sorted[:mu]
                diff = (yw - x) / sigma
                delta = (mean_new - x) / sigma
                pc = (1 - c_c) * pc + c_c * delta
                outer = np.outer(pc, pc)
                C = (1 + c_1 + 2 * c_mu - c_c) * C + c_1 * (outer - C) + c_mu * np.sum(
                    w[:, None, None] * diff[:, None, :] * diff[:, None, :], axis=0
                )

                # Recompute eigendecomposition
                ev, V = np.linalg.eigh(C + 1e-20 * np.eye(n))
                ev = np.maximum(ev, 1e-20)
                D = np.sqrt(ev)
                B = V
                invsqrtC = V @ np.diag(1.0 / np.sqrt(ev)) @ V.T

                x = mean_new
                fx = fy_sorted[0]

                # Termination conditions
                if sigma < 1e-10:
                    break
                if stall_count > 15:
                    break

            return x, fx

        # Allocate 60% of budget to CMA-ES
        cma_budget = int(budget * 0.6)
        x_end, f_end = cma_es_phase(best_x.copy(), best_f, cma_budget)

        # Update best
        if f_end < best_f:
            best_f, best_x = f_end, x_end.copy()

        # ---- Phase 2: (1+1)-ES with 1/5 success rule ----
        def es_phase(x0, f0, budget_alloc):
            nonlocal best_x, best_f, calls
            n = dim
            x = x0.copy()
            fx = f0
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            sigma = 0.5  # start with moderate step
            success_count = 0
            total_trials = 0
            no_improve = 0

            while calls < budget_alloc and time_ok():
                z = rng.standard_normal(n)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = F(y)
                total_trials += 1

                if fy < fx:
                    x = y.copy()
                    fx = fy
                    success_count += 1
                    no_improve = 0
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    no_improve += 1

                # 1/5 success rule
                if total_trials >= 5:
                    if success_count / total_trials > 1.0 / 5:
                        sigma *= 1.25
                    else:
                        sigma *= 0.8
                    success_count = 0
                    total_trials = 0

                sigma = np.clip(sigma, 1e-12, 10.0)

                # Escape from local minima: if stuck, add random perturbation
                if no_improve > 20:
                    x = x + rng.standard_normal(n) * 0.3
                    x = np.clip(x, -5, 5)
                    fx = F(x)
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    no_improve = 0

            return x, fx

        es_budget = int(budget * 0.3)
        x_end2, f_end2 = es_phase(best_x.copy(), best_f, es_budget)

        if f_end2 < best_f:
            best_f, best_x = f_end2, x_end2.copy()

        # ---- Phase 3: Coordinate-descent polish ----
        def coord_descent_polish(x0, f0, budget_alloc):
            nonlocal best_x, best_f, calls
            n = dim
            x = x0.copy()
            fx = f0
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            # Adaptive step sizes per dimension
            steps = np.full(n, 0.1)
            min_step = 1e-8

            while calls < budget_alloc and time_ok():
                improved = False
                for d in range(n):
                    if calls >= budget_alloc or not time_ok():
                        break
                    # Try positive and negative directions
                    for sign in [1, -1]:
                        if calls >= budget_alloc or not time_ok():
                            break
                        y = x.copy()
                        y[d] += sign * steps[d]
                        y = np.clip(y, -5, 5)
                        fy = F(y)
                        if fy < fx:
                            x = y.copy()
                            fx = fy
                            improved = True
                            if fx < best_f:
                                best_f, best_x = fx, x.copy()
                        else:
                            # Reduce step for this dimension
                            steps[d] *= 0.7

                # If no improvement in full cycle, shrink all steps
                if not improved:
                    steps *= 0.5
                    if np.all(steps < min_step):
                        break
                    # Add small random perturbation to escape
                    if np.random.random() < 0.3:
                        d_rand = rng.integers(n)
                        x[d_rand] += rng.standard_normal() * 0.05
                        x = np.clip(x, -5, 5)
                        fx = F(x)
                        if fx < best_f:
                            best_f, best_x = fx, x.copy()

            return x, fx

        # Use remaining budget for coordinate descent
        remaining = budget - calls
        if remaining > 10 and time_ok():
            x_end3, f_end3 = coord_descent_polish(best_x.copy(), best_f, remaining)
            if f_end3 < best_f:
                best_f, best_x = f_end3, x_end3.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
