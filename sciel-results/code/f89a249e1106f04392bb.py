"""Model-and-optimize: CMA-ES with adaptive restarts for BBOB f11 Discus."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for BBOB-style Discus function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)

        best_x = None
        best_f = float('inf')
        call_count = 0

        def try_eval(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def cma_es_restart(x0, sigma0, max_evals, restart_threshold=100):
            """Run CMA-ES with adaptive restart based on stagnation."""
            nonlocal call_count, best_f, best_x, deadline

            n = dim
            lambda_ = 4 + int(3 * np.log(n))
            mu = lambda_ // 2
            weights = np.log(np.arange(mu, 0, -1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (n + mueff + 5)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_c = min(c_c, 1.0)
            mueff_inv = 1.0 / mueff
            c_1 = (2 - mueff_inv + c_sigma) / ((n + 1.3) ** 2 + mueff_inv + c_sigma)
            c_mu = (2 * (mueff_inv - 1) + c_sigma) / ((n + 2) ** 2 + mueff_inv + c_sigma)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff_inv - 1) / (n + 1)) - 1) + c_sigma

            B = np.eye(n)
            D = np.ones(n)
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            x_mean = x0.copy()
            sigma = sigma0
            evals = 0
            best_in_run = best_f
            last_improvement_eval = 0

            # Initial population
            pop_x = []
            pop_f = []
            for i in range(lambda_):
                z = rng.standard_normal(n)
                xi = x_mean + sigma * (B @ (D * z))
                fi = try_eval(xi)
                if fi is None:
                    break
                pop_x.append(xi)
                pop_f.append(fi)
                evals += 1
                if fi < best_in_run:
                    best_in_run = fi
                    last_improvement_eval = evals
                if evals >= max_evals:
                    break

            if len(pop_x) < 2:
                return

            idx = np.argsort(pop_f)
            pop_x = [pop_x[i] for i in idx]
            pop_f = [pop_f[i] for i in idx]

            while evals < max_evals and call_count < budget:
                if time.time() > deadline - 5:
                    break

                # Check for stagnation and potential restart
                if evals - last_improvement_eval > restart_threshold:
                    # Try a small perturbation from current best
                    if call_count < budget:
                        x_perturb = best_x + 0.5 * rng.standard_normal(dim)
                        fp = try_eval(x_perturb)
                        if fp is not None and fp < best_f - 1e-10:
                            # Found improvement, reset stagnation counter
                            last_improvement_eval = evals
                            x_mean = best_x.copy()
                            sigma = 0.5
                            # Regenerate population
                            pop_x = []
                            pop_f = []
                            for i in range(lambda_):
                                z = rng.standard_normal(n)
                                xi = x_mean + sigma * (B @ (D * z))
                                fi = try_eval(xi)
                                if fi is None:
                                    break
                                pop_x.append(xi)
                                pop_f.append(fi)
                                evals += 1
                                if fi < best_in_run:
                                    best_in_run = fi
                                if evals >= max_evals:
                                    break
                            if len(pop_x) < 2:
                                break
                            idx = np.argsort(pop_f)
                            pop_x = [pop_x[i] for i in idx]
                            pop_f = [pop_f[i] for i in idx]
                            continue
                    else:
                        break

                # Update weights
                w = weights[:min(mu, len(pop_f))]
                x_mean_new = np.sum(np.array(pop_x[:len(w)]) * w[:, None], axis=0)

                # Update translation (cm)
                pc = (1 - c_c) * pc + c_c * np.sqrt(mueff) / (sigma * np.sqrt(np.sum(D ** 2))) * (x_mean_new - x_mean)
                hs = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (evals + 1))) / chi_n < 1.4 + 2 / (n + 1)
                C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - hs) * c_c * (2 - c_c) * C) + \
                    c_mu * np.sum(w[:, None, None] * (np.array(pop_x[:len(w)]) - x_mean) @ (np.array(pop_x[:len(w)]) - x_mean).T, axis=0)

                # Eigendecomposition
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-10))
                B = B * np.sqrt(D[:, None])

                # Update sigma (cs)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / (sigma * np.sqrt(np.sum(D ** 2))) * (x_mean_new - x_mean)
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = min(sigma, 1e6)
                sigma = max(sigma, 1e-12)

                x_mean = x_mean_new

                # Generate new population
                pop_x_new = []
                pop_f_new = []
                for i in range(lambda_):
                    z = rng.standard_normal(n)
                    xi = x_mean + sigma * (B @ z)
                    fi = try_eval(xi)
                    if fi is None:
                        break
                    pop_x_new.append(xi)
                    pop_f_new.append(fi)
                    evals += 1
                    if fi < best_in_run:
                        best_in_run = fi
                        last_improvement_eval = evals
                    if evals >= max_evals:
                        break

                if len(pop_x_new) < 2:
                    break

                idx_new = np.argsort(pop_f_new)
                pop_x = [pop_x_new[i] for i in idx_new]
                pop_f = [pop_f_new[i] for i in idx_new]

        # Phase 1: Dense radial sweep to find the narrow axis of the Discus function
        # The Discus function has extreme anisotropy - one narrow axis
        n_dirs = 150
        n_radii = 12

        # Generate directions: mix of axis-aligned and random
        directions = []
        # Axis-aligned directions (positive and negative)
        for i in range(dim):
            for sign in [-1, 1]:
                d = np.zeros(dim)
                d[i] = sign
                directions.append(d)
        # Random directions
        while len(directions) < n_dirs:
            d = rng.standard_normal(dim)
            norm = np.linalg.norm(d)
            if norm > 1e-10:
                directions.append(d / norm)

        radii = np.geomspace(0.1, 4.0, n_radii)

        for d in directions:
            for r in radii:
                if call_count >= budget:
                    break
                x = d * r
                try_eval(x)
            if call_count >= budget:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            call_count += 1

        # Phase 2: CMA-ES with restarts
        if call_count < budget:
            x0 = best_x.copy()
            sigma0 = 1.0
            cma_es_restart(x0, sigma0, budget - call_count)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            call_count += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
