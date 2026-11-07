"""Portfolio optimizer for BBOB f19 (Griewank-Rosenbrock)."""
import numpy as np
import time


class Optimizer:
    """Multi-start portfolio optimizer with adaptive strategies."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_x = None
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            fx = float(fx)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return x, fx

        def local_search(x0, step=0.5, max_iter=20):
            """Simple coordinate descent local search."""
            nonlocal calls_made, best_x, best_f
            x = x0.copy()
            _, f_cur = safe_call(x)
            for it in range(max_iter):
                if calls_made >= budget or time.time() >= deadline:
                    break
                improved = False
                for i in range(dim):
                    if calls_made >= budget or time.time() >= deadline:
                        break
                    for direction in [1, -1]:
                        x_try = x.copy()
                        x_try[i] += direction * step
                        x_try = np.clip(x_try, -5.0, 5.0)
                        _, f_try = safe_call(x_try)
                        if f_try < f_cur:
                            x = x_try
                            f_cur = f_try
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 0.01:
                        break
            return x, f_cur

        def run_cma_es(x_init, sigma_init, max_generations):
            """Run a short CMA-ES from given initial state."""
            nonlocal calls_made, best_x, best_f

            lam = 11
            mu = 5
            alpha_mu = 2.0 / (mu + 1)
            alpha_sigma = 0.8 + 0.5 * (mu - 2) / (dim + mu - 2)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + alpha_sigma
            cc = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
            cs = (mu + 2 + alpha_sigma / 2) / (dim + mu + 5)
            c1 = (2 / ((dim + 1.3) ** 2 + mu)) * (2 - alpha_mu / 2)
            cmu = c1 * (mu - 2 + alpha_mu / c1)
            damps = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1) - 1)) + alpha_sigma
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

            x_mean = x_init.copy()
            sigma = sigma_init
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = 1.0 / np.sum(w ** 2)

            for gen in range(max_generations):
                if calls_made >= budget or time.time() >= deadline:
                    break

                # Eigendecomposition for sampling
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

                Y = np.random.default_rng().standard_normal((lam, dim))
                X = x_mean + sigma * (Y @ C_sqrt.T)

                F = []
                for i in range(lam):
                    if calls_made >= budget or time.time() >= deadline:
                        break
                    _, fi = safe_call(X[i])
                    F.append(fi)

                if len(F) < mu:
                    break

                idx = np.argsort(F)
                X_sorted = X[idx[:mu]]

                x_new_mean = np.sum(w[:, None] * X_sorted, axis=0)

                C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_sqrt_inv @ (x_new_mean - x_mean) / sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made // lam)))) < (1.4 + 2 / (dim + 1)) * chi_n

                sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (x_new_mean - x_mean) / sigma
                delta_h = (1 - hsig) * cc * (2 - cc)
                C = (1 - c1 - cmu + delta_h) * C
                C += c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                C += cmu * np.sum(w[:, None, None] * np.outer(X_sorted - x_mean, X_sorted - x_mean), axis=0) / sigma ** 2

                C = (C + C.T) / 2
                eigvals_c, eigvecs_c = np.linalg.eigh(C)
                eigvals_c = np.maximum(eigvals_c, 1e-12)
                C = eigvecs_c @ np.diag(eigvals_c) @ eigvecs_c.T

                x_mean = x_new_mean

                if sigma < 1e-6:
                    break

            return x_mean, sigma

        # Phase 1: Evaluate several diverse starting points
        rng = np.random.default_rng(42)
        start_points = [
            np.zeros(dim),
            rng.uniform(-3, 3, size=dim),
            rng.uniform(-3, 3, size=dim),
            np.array([2.0, -2.0, 1.5, -1.5, 0.5])[:dim],
            np.array([-1.0, 1.0, -2.0, 2.0, 0.0])[:dim],
        ]

        for sp in start_points:
            if calls_made >= budget or time.time() >= deadline:
                break
            _, _ = safe_call(sp)

        # Phase 2: Run multiple short CMA-ES from best starts
        # Sort starts by fitness and pick top ones
        start_results = []
        for sp in start_points:
            if calls_made >= budget or time.time() >= deadline:
                break
            # We already evaluated these, use current best_f tracking
        # Just use the best point found so far and a few random perturbations
        if best_x is not None:
            cma_starts = [best_x.copy()]
            for _ in range(2):
                pert = best_x + rng.normal(0, 1.0, size=dim)
                pert = np.clip(pert, -5, 5)
                cma_starts.append(pert)
        else:
            cma_starts = [np.zeros(dim), rng.uniform(-2, 2, size=dim)]

        for cs in cma_starts:
            if calls_made >= budget or time.time() >= deadline:
                break
            # Run CMA-ES with limited generations
            remaining_budget = budget - calls_made
            max_gens = min(30, max(5, remaining_budget // lam))
            if max_gens < 5:
                break
            run_cma_es(cs, 1.0, max_gens)

        # Phase 3: Local refinement from the best point
        if calls_made < budget and time.time() < deadline and best_x is not None:
            local_search(best_x, step=0.3, max_iter=30)

        # Phase 4: If still have budget, try another round of short CMA-ES
        if calls_made < budget * 0.7 and time.time() < deadline and best_x is not None:
            remaining = budget - calls_made
            if remaining > 50:
                max_gens = min(20, remaining // lam)
                if max_gens >= 5:
                    run_cma_es(best_x, 0.5, max_gens)
                    if calls_made < budget and time.time() < deadline:
                        local_search(best_x, step=0.1, max_iter=20)

        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f
