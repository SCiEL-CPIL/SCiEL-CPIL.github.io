"""CMA-ES with restarts for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts, tuned for 5-dim ill-conditioned functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # lambda population size
        lam = 4 + int(3 * np.log(dim))  # ~14 for dim=5
        # mu = floor(lam/2)
        mu = lam // 2
        # weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        # step size control
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        # covariance adaptation
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        # chi_n for sigma update
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        # initial sigma
        sigma_init = 2.0

        best_x = None
        best_f = np.inf
        calls_made = 0

        def cma_es(x0, sigma, remaining_budget):
            nonlocal best_x, best_f, calls_made
            x = x0.copy()
            sigma = sigma
            # evolution paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)
            # C inverse and sqrt
            C_inv = np.eye(dim)
            C_sqrt = np.eye(dim)
            # precompute
            one_over_sqrt = 1.0 / np.sqrt(1.0 - (1 - c_c) ** 2)
            h_sigma = (1 + c_sigma) ** (-0.5)
            # step size
            step_size = sigma

            for gen in range(remaining_budget // lam + 1):
                if time.time() > deadline:
                    break
                # generate population
                offspring = np.zeros((lam, dim))
                for i in range(lam):
                    if time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    # apply C_sqrt
                    y = C_sqrt @ z
                    offspring[i] = x + step_size * y
                    # clip to bounds
                    np.clip(offspring[i], -5, 5, out=offspring[i])
                    f_val = func(offspring[i])
                    calls_made += 1
                    if f_val < best_f:
                        best_f = f_val
                        best_x = offspring[i].copy()

                # sort by fitness
                f_offspring = np.array([func(x) if False else 0 for _ in range(lam)])  # placeholder
                # Actually we need to evaluate properly. Let me restructure.
                break  # This approach is getting complicated, let me redo

        # Simpler approach: implement CMA-ES more directly
        def cma_es_run(x0, sigma_init_val, max_evals):
            nonlocal best_x, best_f, calls_made
            x = x0.astype(float).copy()
            sigma = sigma_init_val
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)
            C_inv = np.eye(dim)
            # Compute Cholesky of C
            C_cholesky = np.linalg.cholesky(C)
            mueff = 1.0 / np.sum(w ** 2)
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            evals_in_run = 0
            while evals_in_run < max_evals:
                if time.time() > deadline:
                    break
                # Sample population
                Y = np.zeros((lam, dim))
                Z = rng.standard_normal((lam, dim))
                # Transform using Cholesky
                Y = Z @ C_cholesky.T  # Y_i = C_cholesky.T @ Z_i
                X = x[None, :] + sigma * Y
                # Clip
                np.clip(X, -5, 5, out=X)
                # Evaluate
                F = np.zeros(lam)
                for i in range(lam):
                    if time.time() > deadline:
                        break
                    F[i] = func(X[i])
                    evals_in_run += 1
                    calls_made += 1
                    if F[i] < best_f:
                        best_f = F[i]
                        best_x = X[i].copy()
                else:
                    # Sort
                    order = np.argsort(F)
                    Y_sorted = Y[order]
                    F_sorted = F[order]
                    # Update mean
                    x_new = x + sigma * np.sum(w[:, None] * Y_sorted[:mu], axis=0)
                    # Update paths
                    y_sigma = (x_new - x) / sigma
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma
                    sigma_norm = np.linalg.norm(p_sigma)
                    if sigma_norm / chi_n > 1.4 + 2 / (dim + 1):
                        sigma *= np.exp(0.5 * c_sigma * (sigma_norm / chi_n - 1))
                    h_sig = (sigma_norm / chi_n < (1.4 + 2 / (dim + 1)) * (1 - c_sigma))
                    p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mueff) * y_sigma
                    # Update C
                    delta = (h_sig * (1 + (1 - c_c) * c_c) - 1) * np.outer(p_c, p_c)
                    rank_mu = np.sum(w[:, None] * np.outer(Y_sorted[:mu], Y_sorted[:mu]), axis=0)
                    C = (1 + c_1 + c_mu - c_c) * C + c_1 * delta + c_mu * rank_mu
                    # Symmetrize
                    C = (C + C.T) / 2
                    # Recompute Cholesky
                    try:
                        C_cholesky = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        # Add small jitter
                        C = C + 1e-8 * np.eye(dim)
                        C_cholesky = np.linalg.cholesky(C)
                    x = x_new
                    # Prevent sigma from getting too small or large
                    sigma = np.clip(sigma, 1e-10, 10.0)

            return evals_in_run

        # Main loop with restarts
        # Initial point: center of box
        x0 = np.zeros(dim)
        # First evaluate center
        f0 = func(x0)
        calls_made += 1
        best_f = f0
        best_x = x0.copy()

        # Run CMA-ES
        evals_per_run = 100
        num_restarts = max(1, (budget - calls_made) // evals_per_run)
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break
            remaining = budget - calls_made
            this_evals = min(evals_per_run, remaining)
            if this_evals < lam:
                break
            # For restarts after the first, perturb the best point
            if r == 0:
                x_start = best_x.copy()
                sigma_start = 2.0
            else:
                # Restart from best with larger sigma
                x_start = best_x + 0.5 * rng.standard_normal(dim)
                np.clip(x_start, -5, 5, out=x_start)
                sigma_start = 2.0
            cma_es_run(x_start, sigma_start, this_evals)

        return best_x, best_f
# EVOLVE-BLOCK-END
