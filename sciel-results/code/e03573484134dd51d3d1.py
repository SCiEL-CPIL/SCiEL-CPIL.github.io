"""CMA-ES + (1+1)-ES hybrid optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES + (1+1)-ES for BBOB f11 (Discus).
    CMA-ES handles the anisotropic discus landscape; (1+1)-ES polishes.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 325  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_check():
            return time.time() > deadline

        # --- Phase 1: Quick exploration to find good starting points ---
        n_explore = min(30, budget // 10)
        points = []
        values = []

        # Center
        p0 = np.zeros(dim)
        f0 = func(np.clip(p0, -5, 5))
        points.append(p0)
        values.append(f0)
        evals = 1

        # Random points
        for _ in range(n_explore - 1):
            if evals >= budget or time_check():
                break
            p = rng.uniform(-4.5, 4.5, dim)
            fp = func(np.clip(p, -5, 5))
            points.append(p)
            values.append(fp)
            evals += 1

        if evals >= budget or time_check():
            best_idx = int(np.argmin(values))
            return points[best_idx], values[best_idx]

        best_idx = int(np.argmin(values))
        best_x = np.array(points[best_idx], dtype=np.float64)
        best_f = values[best_idx]

        # --- Phase 2: CMA-ES with restarts ---
        # CMA-ES parameters
        # For 5D, lambda = 4 + floor(3 * log(5)) = 4 + 3 = 7
        lam = 4 + int(3 * np.log(dim))
        
        # Reserve budget for final polish
        polish_budget = min(60, budget // 8)
        cma_budget = budget - polish_budget

        # Number of CMA-ES restarts
        n_restarts = 3
        budget_per_restart = cma_budget // n_restarts

        for restart in range(n_restarts):
            if evals >= cma_budget or time_check():
                break

            # Determine starting point
            if restart == 0:
                x_mean = best_x.copy()
                sigma = 1.0
            else:
                # Pick one of the top exploration points
                top_k = min(3, len(points))
                top_indices = np.argsort(values)[:top_k]
                x_mean = np.array(points[top_indices[restart % top_k]], dtype=np.float64)
                sigma = 0.5

            # CMA-ES state
            x_mean = np.clip(x_mean, -5, 5)
            sigma = max(sigma, 0.1)
            C = np.eye(dim)
            C_inv_sqrt = np.eye(dim)  # For simplicity, use identity initially
            pc = np.zeros(dim)
            cs = (lam // 2 + 1) / (dim + 2 + lam // 2)
            c1 = 2 / ((dim + 1.3) ** 2 + lam)
            cmu = min(1 - c1, 2 / ((dim + 2) ** 2 + lam))
            cc = (4 + dim / lam) / (dim + 4 + 2 * lam / dim)
            damps = 1 + 2 * max(0, np.sqrt((lam - 2) / (dim + 2) - 1)) + cc

            # Initial population
            pop_x = []
            pop_f = []
            for i in range(lam):
                if evals >= cma_budget or time_check():
                    break
                z = rng.standard_normal(dim)
                xi = x_mean + sigma * z
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                pop_x.append(xi)
                pop_f.append(fi)
                evals += 1

            if len(pop_x) < lam:
                break

            # CMA-ES main loop
            max_iter = budget_per_restart // lam
            for it in range(max_iter):
                if evals >= cma_budget or time_check():
                    break

                # Sort by fitness
                order = np.argsort(pop_f)
                pop_x = [pop_x[i] for i in order]
                pop_f = [pop_f[i] for i in order]

                # Update best
                if pop_f[0] < best_f:
                    best_f = pop_f[0]
                    best_x = pop_x[0].copy()

                # Weighted mean
                weights = np.log(lam / 2 + 1) - np.arange(lam)
                weights /= weights.sum()
                mu_eff = 1.0 / np.sum(weights[:lam // 2] ** 2)

                x_new_mean = np.zeros(dim)
                for i in range(lam):
                    x_new_mean += weights[i] * pop_x[i]
                x_mean = x_new_mean

                # Update evolution path
                pc = (1 - cs) * pc + cs / np.sqrt(mu_eff) * (x_mean - x_new_mean) / sigma
                if np.linalg.norm(pc) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1))) > (1.4 + 2 / (dim + 1)) * damps:
                    c1 = c1 * 0.5
                    cmu = cmu * 0.5

                # Update covariance
                diffmat = np.array([(pop_x[i] - x_new_mean) / sigma for i in range(lam)])
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (cs * (2 - cs)) * C) \
                    + cmu / mu_eff * np.sum([np.outer(diffmat[i], diffmat[i]) for i in range(lam)], axis=0)

                # Ensure C is symmetric positive definite
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update sigma
                sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(pc) / np.sqrt(mu_eff) - 1))
                sigma = max(sigma, 1e-10)

                # Generate new population
                new_pop_x = []
                new_pop_f = []
                for i in range(lam):
                    if evals >= cma_budget or time_check():
                        break
                    z = rng.standard_normal(dim)
                    # Transform using C
                    xi = x_mean + sigma * (eigvecs @ (np.sqrt(eigvals) * z))
                    xi = np.clip(xi, -5, 5)
                    fi = func(xi)
                    new_pop_x.append(xi)
                    new_pop_f.append(fi)
                    evals += 1

                if len(new_pop_x) < lam:
                    break

                pop_x = new_pop_x
                pop_f = new_pop_f

                # Stopping criterion: if sigma is too small or no improvement
                if sigma < 1e-8:
                    break

        # --- Phase 3: (1+1)-ES Local Polish ---
        x_local = best_x.copy()
        f_local = best_f
        sigma_local = 0.1

        no_improve = 0
        max_no_improve = 30

        while evals < budget:
            if time_check():
                break

            z = rng.standard_normal(dim)
            y = x_local + sigma_local * z
            y = np.clip(y, -5, 5)
            fy = func(y)
            evals += 1

            if fy < f_local:
                x_local = y
                f_local = fy
                sigma_local = max(sigma_local * 1.08, 1e-6)
                no_improve = 0
            else:
                sigma_local = max(sigma_local * 0.92, 1e-6)
                no_improve += 1

            if no_improve >= max_no_improve:
                break

            if f_local < best_f:
                best_f = f_local
                best_x = x_local.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
