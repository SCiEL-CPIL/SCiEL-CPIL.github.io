import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 270  # 30s margin for 300s limit

        best_x = None
        best_f = np.inf
        evals_used = 0

        def eval_point(x):
            nonlocal best_x, best_f, evals_used
            if evals_used >= budget:
                return best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            evals_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global Initialization - use 20 points to find a good starting basin
        # f7 is step ellipsoidal: the function is flat in one direction and
        # steep in others. We need to find the right region.
        rng = np.random.default_rng(42)
        init_candidates = []
        for i in range(20):
            if evals_used >= budget:
                break
            if i < 8:
                x = rng.uniform(-1.5, 1.5, dim)
            elif i < 14:
                x = rng.uniform(-3.0, 3.0, dim)
            else:
                x = rng.uniform(-4.5, 4.5, dim)
            f = eval_point(x)
            init_candidates.append((f, x.copy()))

        init_candidates.sort(key=lambda k: k[0])
        x_start = init_candidates[0][1].copy()

        # Helper: run CMA-ES with adaptive restarts
        def run_cma_es(mean_init, sigma_init, max_evals, seed_offset=0, stagnation_window=15):
            nonlocal evals_used, best_x, best_f
            rng_local = np.random.default_rng(42 + seed_offset)

            lam = max(4, int(4 + 3 * np.log(dim)))
            mu = max(1, lam // 2)
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mueff = 1.0 / np.sum(w**2)

            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = (2 - mueff / (dim + 2) + c_c) / ((dim + 1.3) ** 2 + mueff + c_c)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

            sigma = sigma_init
            mean = mean_init.copy()
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            L = np.eye(dim)

            cma_evals = 0
            best_in_run = np.inf
            no_improve_count = 0
            restart_count = 0

            while cma_evals + lam <= max_evals and evals_used < budget:
                if time.time() > deadline:
                    break

                # Generate offspring
                ys = []
                xs = []
                fs = []

                for i in range(lam):
                    if evals_used >= budget or time.time() > deadline:
                        break
                    z = rng_local.standard_normal(dim)
                    y = L @ z
                    x_new = mean + sigma * y
                    f_val = eval_point(x_new)

                    ys.append(y)
                    xs.append(x_new)
                    fs.append(f_val)

                if len(xs) < lam:
                    break

                cma_evals += lam

                ys_arr = np.array(ys)
                xs_arr = np.array(xs)
                fs_arr = np.array(fs)

                # Sort by fitness
                idx = np.argsort(fs_arr)
                xs_sorted = xs_arr[idx]
                ys_sorted = ys_arr[idx]
                fs_sorted = fs_arr[idx]

                # Track improvement
                if fs_sorted[0] < best_in_run - 1e-10:
                    best_in_run = fs_sorted[0]
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Update mean
                mean_old = mean.copy()
                mean = np.sum(w[:mu, None] * xs_sorted[:mu], axis=0)

                # Compute C_inv_sqrt for path updates
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                # Update evolution paths
                diff = (mean - mean_old) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ diff)
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c)) * (C_inv_sqrt @ diff)

                # Update sigma
                norm_ps = np.linalg.norm(p_sigma)
                chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2))
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))

                # Update covariance
                sum_w_y = np.zeros((dim, dim))
                for i in range(mu):
                    sum_w_y += w[i] * np.outer(ys_sorted[i], ys_sorted[i])

                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c)**2) * C) + c_mu * sum_w_y
                C = (C + C.T) / 2
                C += 1e-12 * np.eye(dim)

                # Update L
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-10 * np.eye(dim)
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C = np.eye(dim)
                        L = np.eye(dim)

                # Stop if converged
                if sigma < 1e-8:
                    break

                # Stagnation-based restart: if no improvement for many generations,
                # restart from best point with larger sigma to escape local basin
                if no_improve_count >= stagnation_window:
                    if restart_count < 3 and cma_evals + lam * 2 <= max_evals:
                        # Restart: increase sigma and reset some state
                        restart_count += 1
                        sigma = max(sigma * 3.0, 0.5)
                        C = np.eye(dim)
                        p_sigma = np.zeros(dim)
                        p_c = np.zeros(dim)
                        L = np.eye(dim)
                        no_improve_count = 0
                        best_in_run = fs_sorted[0]
                    else:
                        break

            return mean, sigma

        # Phase 2: First CMA-ES run
        cma1_budget = min(300, budget - evals_used)
        if cma1_budget > 20:
            _, sigma_final = run_cma_es(x_start, 2.0, cma1_budget, seed_offset=0, stagnation_window=12)

        # Phase 3: If budget remains, do a second CMA-ES run from best point
        remaining = budget - evals_used
        if remaining > 15 and time.time() < deadline:
            sigma_restart = max(sigma_final * 0.3, 0.05)
            run_cma_es(best_x.copy(), sigma_restart, remaining, seed_offset=500, stagnation_window=10)

        return best_x, best_f
