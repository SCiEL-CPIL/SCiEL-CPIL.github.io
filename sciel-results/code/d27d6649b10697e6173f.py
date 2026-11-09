"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-restart and adaptive exploration for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track global best
        x_best = None
        f_best = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal x_best, f_best, calls_made
            fx = func(x)
            calls_made += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        def run_cmaes(x0, sigma0, max_calls, rng_local):
            nonlocal calls_made
            lambda_ = max(4 * dim, 20)
            mu = lambda_ // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1 + 2 * max(0, (mueff - 2) / (dim + mueff)) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            gen = 0
            local_calls = 0

            while local_calls + lambda_ <= max_calls and time.time() < deadline and calls_made < budget:
                # Generate population
                B = rng_local.standard_normal((lambda_, dim))
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                    C = np.eye(dim)
                Y = x_mean[None, :] + sigma * (B @ L)
                Y = np.clip(Y, -5, 5)

                # Evaluate
                F = []
                for y in Y:
                    if calls_made >= budget:
                        break
                    F.append(evaluate(y))
                    local_calls += 1

                if len(F) < lambda_:
                    return

                F = np.array(F)
                gen += 1

                # Sort
                order = np.argsort(F)
                Y_sorted = Y[order]
                F_sorted = F[order]

                # Update mean
                x_old = x_mean.copy()
                x_mean = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)

                # Update step size
                try:
                    L_inv = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    L_inv = np.eye(dim)
                step = (L_inv @ (x_mean - x_old).T).T[0]
                step_norm = np.linalg.norm(step)
                if step_norm > 1e-16:
                    ps = (1 - c_sigma) * ps + c_sigma * step_norm / chi_n * (step / step_norm)
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                    sigma = np.clip(sigma, 1e-10, 50.0)

                # Update covariance matrix
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * gen))) < (1.4 + 2 / (dim + 1)) * chi_n
                pc = (1 - c_c) * pc + hsig * c_c * (x_mean - x_old) / max(sigma, 1e-10)

                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                rank_1 = np.outer(pc, pc)
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    y_i = (Y_sorted[i] - x_old) / max(sigma, 1e-10)
                    rank_mu += weights[i] * np.outer(y_i, y_i)

                C = (1 + c_1 * (dim + 1 - c_1) - c_mu - c_c * (1 - c_c) * delta_hsig) * C \
                    + c_1 * (rank_1 - C) + c_mu * rank_mu

                # Ensure C is positive definite
                C = (C + C.T) / 2
                try:
                    eigvals = np.linalg.eigvalsh(C)
                    if np.min(eigvals) < 1e-8:
                        C = C + 1e-8 * np.eye(dim)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)

                # Convergence check
                if sigma < 1e-8 or (gen > 50 and np.ptp(F_sorted[:mu]) < 1e-6):
                    break

            # Use remaining budget for (1+1) if any
            remaining = max_calls - local_calls
            if remaining > 0 and x_best is not None:
                x = x_best.copy()
                fx = f_best
                s = max(sigma, 0.1)
                for _ in range(remaining):
                    if time.time() > deadline or calls_made >= budget:
                        break
                    y = np.clip(x + s * rng_local.standard_normal(dim), -5, 5)
                    fy = evaluate(y)
                    if fy < fx:
                        x, fx = y, fy
                        s = min(s * 1.1, 5.0)
                    else:
                        s = max(s * 0.9, 1e-6)

            return x_mean, f_best

        def run_1plus1(x0, max_calls, rng_local, sigma_init=1.0):
            nonlocal calls_made
            x = x0.copy()
            fx = evaluate(x)
            sigma = sigma_init
            for _ in range(max_calls - 1):
                if time.time() > deadline or calls_made >= budget:
                    break
                y = np.clip(x + sigma * rng_local.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.1, 5.0)
                else:
                    sigma = max(sigma * 0.9, 1e-6)
            return x, fx

        # Strategy: 3 phases
        phase1_budget = int(budget * 0.55)
        phase2_budget = int(budget * 0.30)
        phase3_budget = budget - phase1_budget - phase2_budget

        # Phase 1: CMA-ES with broad initial exploration
        x0 = rng.uniform(-4, 4, dim)
        sigma0 = 2.5
        rng_local = np.random.default_rng(123)
        run_cmaes(x0, sigma0, phase1_budget, rng_local)

        # Phase 2: Local search around best with multiple restarts
        if x_best is not None and calls_made < budget:
            n_local_restarts = 3
            evals_per_local = phase2_budget // n_local_restarts
            for lr in range(n_local_restarts):
                if time.time() > deadline or calls_made >= budget:
                    break
                if lr == 0:
                    x_start = x_best.copy()
                    sigma_start = 0.5
                else:
                    x_start = np.clip(x_best + rng.standard_normal(dim) * 1.5, -5, 5)
                    sigma_start = 1.0
                rng_local = np.random.default_rng(456 + lr)
                run_1plus1(x_start, min(evals_per_local, budget - calls_made), rng_local, sigma_start)

        # Phase 3: Final fine-grained polish
        if x_best is not None and calls_made < budget:
            x_start = x_best.copy()
            sigma_start = 0.15
            rng_local = np.random.default_rng(789)
            run_1plus1(x_start, min(phase3_budget, budget - calls_made), rng_local, sigma_start)

        return x_best, f_best

# EVOLVE-BLOCK-END
