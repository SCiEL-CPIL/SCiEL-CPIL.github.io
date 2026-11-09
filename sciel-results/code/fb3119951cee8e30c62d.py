"""CMA-ES with adaptive restarts for Rastrigin rotated f15."""
import numpy as np
import time
import math


class Optimizer:
    """CMA-ES with adaptive restarts for Rastrigin rotated in 5D."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls_used = 1

        # Phase 1: Initial exploration with diverse starting points
        # Use 20 starts with mixed ranges for broader coverage
        n_initial = 20
        start_points = []
        start_fvals = []

        # First, evaluate the origin
        start_points.append(best_x.copy())
        start_fvals.append(best_f)

        for i in range(n_initial - 1):
            if time.time() > deadline or calls_used >= budget:
                break
            # Mix of different ranges to cover the space
            if i < 5:
                x = rng.uniform(-4.5, 4.5, dim)
            elif i < 10:
                x = rng.uniform(-3.0, 3.0, dim)
            else:
                x = rng.uniform(-1.5, 1.5, dim)
            fx = func(x)
            calls_used += 1
            start_points.append(x.copy())
            start_fvals.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        remaining_budget = budget - calls_used
        if remaining_budget <= 0:
            return (best_x, best_f)

        # Phase 2: Hierarchical budget allocation
        # Sort starts by function value and allocate budget hierarchically
        # Top candidate gets 1/4, next two get 1/8 each, rest get 1/16 each
        order = np.argsort(start_fvals)
        
        # Allocate budget: top 1 gets 1/4, next 2 get 1/8 each, next 3 get 1/16 each
        allocations = {}
        alloc_idx = 0
        total_alloc = 0
        for rank, idx in enumerate(order):
            if alloc_idx >= 6:  # Only allocate to top 6 starts
                break
            if rank == 0:
                alloc = remaining_budget // 4
            elif rank <= 2:
                alloc = remaining_budget // 8
            else:
                alloc = remaining_budget // 16
            allocations[idx] = alloc
            total_alloc += alloc
            alloc_idx += 1

        # Run CMA-ES with adaptive restarts for each allocated start
        for idx in order:
            if time.time() > deadline or calls_used >= budget:
                break
            if idx not in allocations:
                continue
            alloc = allocations[idx]
            if alloc < 20:
                continue
                
            x_start = start_points[idx]
            f_start = start_fvals[idx]
            
            x_s, f_s, calls_used = self._cma_es_restart(
                func, x_start, f_start, alloc, dim, rng, deadline, calls_used
            )
            if f_s < best_f:
                best_f = f_s
                best_x = x_s.copy()

        # Phase 3: Final refinement with small sigma
        if time.time() < deadline and calls_used < budget:
            fine_sigma = 0.1
            for i in range(min(30, budget - calls_used)):
                if time.time() > deadline:
                    break
                x = best_x + rng.normal(0, fine_sigma, dim)
                x = np.clip(x, -4.5, 4.5)
                fx = func(x)
                calls_used += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    fine_sigma *= 0.8

        return (best_x, best_f)

    def _cma_es_restart(self, func, x0, f0, budget, dim, rng, deadline, calls_used):
        """CMA-ES with periodicity-aware restart mechanism."""
        lam = 10
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))

        mean = x0.copy()
        sigma = 1.0
        best_x = x0.copy()
        best_f = f0
        best_f_history = [f0]

        # Restart parameters
        stagnation_window = 3  # Generations without improvement before restart
        max_restarts = 3
        restart_count = 0
        no_improve_count = 0

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)

        def get_chol_inv(C_mat):
            try:
                L = np.linalg.cholesky(C_mat)
                return np.linalg.inv(L)
            except:
                return np.eye(dim)

        C_sqrt_inv = get_chol_inv(C)

        gen = 0
        while calls_used < budget and time.time() < deadline and restart_count < max_restarts:
            # Generate population
            pop = []
            for i in range(lam):
                if time.time() > deadline or calls_used >= budget:
                    break
                z = rng.standard_normal(dim)
                x = mean + sigma * (C_sqrt_inv @ z)
                x = np.clip(x, -4.5, 4.5)
                fx = func(x)
                calls_used += 1
                pop.append((x, fx))
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    no_improve_count = 0

            if len(pop) < mu:
                break

            # Sort by fitness
            pop.sort(key=lambda t: t[1])
            selected = pop[:mu]

            # Update mean
            new_mean = np.zeros(dim)
            for i, (x, fx) in enumerate(selected):
                new_mean += weights[i] * x
            mean = new_mean

            # Update sigma
            if mu_eff > 0 and sigma > 1e-12:
                actual_sigma = np.sqrt(np.mean([
                    np.sum((x - mean)**2) for x, _ in selected
                ])) / np.sqrt(dim)
                if actual_sigma > 0:
                    sigma *= math.exp(c_sigma / d_sigma * (actual_sigma / sigma - 1))

            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * np.mean([
                (x - mean) / max(sigma, 1e-12) for x, _ in selected
            ], axis=0)

            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * np.mean([
                (x - mean) / max(sigma, 1e-12) for x, _ in selected
            ], axis=0)

            # Update covariance matrix
            diff = np.zeros((dim, dim))
            for i, (x, fx) in enumerate(selected):
                d = (x - mean) / max(sigma, 1e-12)
                diff += weights[i] * np.outer(d, d)

            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * np.eye(dim)) + c_mu * diff
            C = (C + C.T) / 2
            C_sqrt_inv = get_chol_inv(C)

            gen += 1
            best_f_history.append(best_f)

            # Check for stagnation and restart
            if len(best_f_history) > stagnation_window:
                recent_best = min(best_f_history[-stagnation_window:])
                if best_f >= recent_best - 1e-8:
                    no_improve_count += 1
                else:
                    no_improve_count = 0

            if no_improve_count >= stagnation_window and restart_count < max_restarts:
                # Restart: reinitialize with larger sigma centered on best incumbent
                restart_count += 1
                mean = best_x.copy()
                sigma = 2.0 * (1 + restart_count * 0.5)  # Increase sigma with each restart
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                C = np.eye(dim)
                C_sqrt_inv = get_chol_inv(C)
                no_improve_count = 0
                best_f_history = [best_f]
                gen = 0

            if sigma < 1e-10:
                break

        return (best_x, best_f, calls_used)
