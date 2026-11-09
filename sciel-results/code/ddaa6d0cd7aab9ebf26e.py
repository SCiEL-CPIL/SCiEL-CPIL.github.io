"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid: diverse initial sampling + CMA-ES with restarts for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        call_count = 0

        def _evaluate(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Diverse initial sampling for early AOCC progress
        n_init = 25
        init_points = []
        init_points.append(np.zeros(dim))
        for i in range(dim):
            for val in [1.0, -1.0, 2.0, -2.0, 3.0, -3.0]:
                pt = np.zeros(dim)
                pt[i] = val
                init_points.append(pt)
        while len(init_points) < n_init:
            init_points.append(rng.uniform(-4, 4, dim))

        for pt in init_points:
            if call_count >= budget - 50:
                break
            if time.time() > deadline:
                break
            _evaluate(pt)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = _evaluate(best_x)

        # Phase 2: CMA-ES with restarts
        lambda_ = 12
        mu_ = lambda_ // 2

        w = np.zeros(mu_)
        for i in range(mu_):
            w[i] = np.log(mu_ + 0.5) - np.log(i + 1)
        w /= w.sum()

        mu_eff = 1.0 / np.sum(w ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * min(mu_ / (dim + 2) - 1, 1) + c_sigma
        c_c = (4 + dim / c_sigma) / (dim + 4 + 2 / c_sigma)
        c_1 = 2 / (dim ** 2 + 6)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / (dim ** 2 + 2))

        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        max_stagnation = 15
        max_restarts = 3
        restart_count = 0

        for restart in range(max_restarts + 1):
            if call_count >= budget or time.time() > deadline:
                break

            # Initialize CMA-ES state
            x_mean = best_x.copy() if restart == 0 else best_x + rng.normal(0, 0.5, dim)
            x_mean = np.clip(x_mean, -5.0, 5.0)
            sigma = 1.5 if restart == 0 else 1.0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            eigvals, eigvecs = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(eigvals, 1e-12))
            B = eigvecs

            stagnation = 0
            prev_best = best_f

            while call_count < budget and time.time() < deadline:
                # Generate population
                offspring = []
                for i in range(lambda_):
                    if call_count >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (B @ (D * z))
                    y = np.clip(y, -5.0, 5.0)
                    f_y = _evaluate(y)
                    offspring.append((y, f_y, z))

                if len(offspring) < mu_:
                    break

                offspring.sort(key=lambda x: x[1])
                selected = offspring[:mu_]

                old_mean = x_mean.copy()
                x_mean = np.sum([w[i] * y for i, (y, f, z) in enumerate(selected)], axis=0)

                C_inv_sqrt = B @ np.diag(1.0 / np.maximum(D, 1e-10)) @ B.T
                delta_mean = x_mean - old_mean
                p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * (C_inv_sqrt @ delta_mean)

                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * np.sum(
                    [w[i] * z for i, (y, f, z) in enumerate(selected)], axis=0)

                norm_p_sigma = np.linalg.norm(p_sigma)
                sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_sigma) ** 2) * np.eye(dim))
                for i in range(mu_):
                    diff = selected[i][0] - x_mean
                    C += c_mu * w[i] * np.outer(diff, diff)
                C = (C + C.T) / 2

                try:
                    eigvals_C, _ = np.linalg.eigh(C)
                    if np.min(eigvals_C) < 1e-12:
                        C = C + (1e-10 - np.min(eigvals_C)) * np.eye(dim)
                    eigvals, eigvecs = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(eigvals, 1e-12))
                    B = eigvecs
                except:
                    pass

                # Check stagnation
                if best_f < prev_best - 1e-8:
                    stagnation = 0
                    prev_best = best_f
                else:
                    stagnation += 1

                if stagnation > max_stagnation:
                    break

                # Check convergence
                if sigma < 1e-8:
                    break

            restart_count += 1
            if restart >= max_restarts:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
