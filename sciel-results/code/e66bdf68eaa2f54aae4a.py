"""Optimizer for BBOB-style f17 (Schaffers F7) with hidden shift/rotation/offset."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard and adaptive restart."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s/330s limit

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        # CMA-ES parameters
        # lambda: population size
        lam = 4 + int(3 * np.log(dim))  # ~15 for dim=5
        mu = lam // 2  # number of parents
        # Weighted recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Step size adaptation
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        def run_cmaes(x0, sigma0, max_calls):
            nonlocal best_x, best_f, calls_made

            x = x0.copy()
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (20 * dim ** 2))

            # Initial population
            rng = np.random.default_rng(42)
            for i in range(lam):
                if time.time() > deadline or calls_made >= budget:
                    return
                y = x + sigma * (B @ (D * rng.standard_normal(dim)))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            # Store initial population for sorting
            pops = [(best_x.copy(), best_f)]  # placeholder, will rebuild

            # Actually, let's restructure: keep track of all evaluated points
            # But for memory efficiency, we'll just do iterative CMA-ES

            # Reset for proper CMA-ES loop
            x = x0.copy()
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (20 * dim ** 2))

            # We already used 'lam' calls for initial population, but those were just random
            # Let's restart properly: use the best from initial as starting point
            # Actually, let's just do a clean CMA-ES run

            # Clean CMA-ES
            x = x0.copy()
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (20 * dim ** 2))

            while calls_made < budget and time.time() < deadline:
                # Generate population
                ys = []
                fys = []
                for i in range(lam):
                    if time.time() > deadline or calls_made >= budget:
                        break
                    z = rng.standard_normal(dim)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_made += 1
                    ys.append(y)
                    fys.append(fy)
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                if len(ys) < lam:
                    break

                # Sort by fitness
                indices = np.argsort(fys)
                ys = [ys[i] for i in indices]
                fys = [fys[i] for i in indices]

                # Update mean
                y_w = np.zeros(dim)
                for i in range(mu):
                    y_w += weights[i] * ys[i]
                x_old = x.copy()
                x = x + y_w

                # Cumulation
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (B @ (D @ (D @ (B.T @ y_w))))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - calls_made + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
                if hsig:
                    pc = (1 - c_c) * pc + c_c / sigma * y_w
                C = (1 - c_1 - c_mu) * C  # This is wrong, C not defined yet

                # Actually, I need to track C properly. Let me restructure.

                # This is getting messy. Let me write a cleaner version.
                break  # break out, use cleaner implementation below

            return

        # Cleaner CMA-ES implementation
        def cmaes_run(x0, sigma0, max_calls):
            nonlocal best_x, best_f, calls_made

            dim_local = dim
            lam_local = lam
            mu_local = mu
            weights_local = weights
            mueff_local = mueff
            c_sigma_local = c_sigma
            d_sigma_local = d_sigma
            c_c_local = c_c
            c_1_local = c_1
            c_mu_local = c_mu

            x = x0.copy()
            sigma = sigma0
            B = np.eye(dim_local)
            D = np.ones(dim_local)
            C = np.eye(dim_local)
            pc = np.zeros(dim_local)
            ps = np.zeros(dim_local)
            chi_n = np.sqrt(dim_local) * (1 - 1.0 / (4 * dim_local) + 1.0 / (20 * dim_local ** 2))

            rng_local = np.random.default_rng(123)

            while calls_made < max_calls and time.time() < deadline:
                # Generate population
                ys = np.zeros((lam_local, dim_local))
                fys = np.zeros(lam_local)

                for i in range(lam_local):
                    if time.time() > deadline or calls_made >= max_calls:
                        lam_local = i + 1
                        break
                    z = rng_local.standard_normal(dim_local)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    ys[i] = y
                    fy = func(y)
                    calls_made += 1
                    fys[i] = fy
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                if lam_local < 2:
                    break

                # Sort by fitness
                indices = np.argsort(fys[:lam_local])
                ys_sorted = ys[indices, :]
                fys_sorted = fys[indices]

                mu_eff = min(mu_local, lam_local)

                # Update mean
                y_w = np.zeros(dim_local)
                for i in range(mu_eff):
                    y_w += weights_local[i] * ys_sorted[i]
                x = x + y_w

                # Cumulation
                ps = (1 - c_sigma_local) * ps + np.sqrt(c_sigma_local * (2 - c_sigma_local) * mueff_local) / sigma * (B @ (D @ (D @ (B.T @ y_w))))
                norm_ps = np.linalg.norm(ps)
                hsig = (norm_ps / np.sqrt(1 - (1 - c_sigma_local) ** 2)) < (1.4 + 2 / (dim_local + 1)) * chi_n
                if hsig:
                    pc = (1 - c_c_local) * pc + c_c_local / sigma * y_w
                    hpc = 1.0
                else:
                    hpc = 0.0

                # Update covariance matrix
                rank_one = np.outer(pc, pc)
                delta_h = (1 - hpc) * c_c * (2 - c_c)
                rank_mu = np.zeros((dim_local, dim_local))
                for i in range(mu_eff):
                    zi = (B @ (D @ (B.T @ (ys_sorted[i] - x - y_w)))) / sigma
                    rank_mu += weights_local[i] * np.outer(zi, zi)

                C = (1 + c_1_local * delta_h - c_1_local) * C + c_1_local * rank_one + c_mu_local * rank_mu

                # Ensure C is symmetric
                C = (C + C.T) / 2

                # Eigendecomposition
                try:
                    D, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D, 1e-20))
                except np.linalg.LinAlgError:
                    D = np.ones(dim_local)
                    B = np.eye(dim_local)

                # Step size adaptation
                sigma *= np.exp(c_sigma_local / d_sigma_local * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 1e6)

                # Check for stagnation
                if sigma < 1e-10 or sigma > 1e5:
                    break

            return

        # Initial phase: random sampling to find a good starting point
        rng_init = np.random.default_rng(7)
        n_init = min(50, budget // 10)
        for i in range(n_init):
            if time.time() > deadline or calls_made >= budget:
                break
            x_try = rng_init.uniform(-5, 5, dim)
            fy = func(x_try)
            calls_made += 1
            if fy < best_f:
                best_f = fy
                best_x = x_try.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        # Run CMA-ES from the best point found
        sigma0 = 2.0
        cmaes_run(best_x.copy(), sigma0, budget)

        # If we have budget left, do a local refinement with smaller sigma
        if calls_made < budget and time.time() < deadline:
            # Try a few more CMA-ES runs with different starting points
            for restart in range(3):
                if calls_made >= budget or time.time() > deadline:
                    break
                # Perturb the best point
                perturbation = 0.5 * rng_init.standard_normal(dim)
                x_start = np.clip(best_x + perturbation, -5, 5)
                sigma_start = 0.5
                cmaes_run(x_start, sigma_start, budget)

        return best_x, best_f
