"""CMA-ES (global) with L-BFGS-B polish for Bent Cigar (f12) optimization."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES for global search on Bent Cigar, followed by L-BFGS-B polish.

    CMA-ES adapts its covariance to the ill-conditioning of f12, while the
    final L-BFGS-B phase exploits gradient-based refinement for convergence.
    A hard deadline guard guarantees we return the best point found well
    before the wall-clock limit, and the evaluation budget is never exceeded.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 275.0          # safety margin vs 300s / 330s kill
        n = self.dim
        budget = self.budget

        # ---- CMA-ES parameters (standard, tuned for small n) ----
        lam = max(4, int(4 + 3 * np.log(n)))
        mu = lam // 2
        w = np.log(np.arange(mu, 0, -1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        d_sigma = 1 + 2 * max(0.0, np.sqrt((mu_eff - 1) / (n + 1)) - 1.0) + c_sigma
        c_cov_pos = (2 + 2 * mu_eff / n) / (n + 3 + 2 * mu_eff / n)
        c_cov_neg = (2 + mu_eff / n) / (n + 3 + 2 * mu_eff / n)

        # Global best (shared across phases)
        best_x = np.zeros(n)
        best_fx = func(best_x)
        calls_used = 1

        def cma_es_phase(x0, sigma0, max_calls):
            """Run CMA-ES from x0 with initial sigma0, using <= max_calls evals."""
            nonlocal best_x, best_fx, calls_used

            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            D, B = np.linalg.eigh(C)
            D = np.maximum(D, 1e-12)
            C_inv_sqrt = B @ np.diag(1.0 / np.sqrt(D)) @ B.T

            no_improve = 0
            stagnation_threshold = 12
            prev_best = best_fx

            for gen in range(1, 5000):
                if time.time() > deadline:
                    break
                if calls_used + lam > max_calls:
                    rem = max_calls - calls_used
                    if rem < 1:
                        break
                    for _ in range(rem):
                        if time.time() > deadline:
                            break
                        z = rng.standard_normal(n)
                        y = x_mean + sigma * (C_inv_sqrt @ z)
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        calls_used += 1
                        if fy < best_fx:
                            best_fx = fy
                            best_x = y.copy()
                    break

                # Sample the population
                ys = np.empty((lam, n))
                fs = np.empty(lam)
                for i in range(lam):
                    z = rng.standard_normal(n)
                    ys[i] = x_mean + sigma * (C_inv_sqrt @ z)
                    ys[i] = np.clip(ys[i], -5, 5)
                    fs[i] = func(ys[i])
                    calls_used += 1
                    if fs[i] < best_fx:
                        best_fx = fs[i]
                        best_x = ys[i].copy()

                # Stagnation signal (population-wide)
                if best_fx < prev_best - 1e-10:
                    no_improve = 0
                else:
                    no_improve += 1
                prev_best = best_fx

                # Recombination
                idx = np.argsort(fs)
                x_mean_new = np.sum(w[:, None] * ys[idx][:mu], axis=0)

                # Evolution paths
                y_vec = (x_mean_new - x_mean) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) \
                    < (1.4 + 2 / (n + 1)) * chi_n
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec

                # Covariance update
                delta_h = (1 - hsig) * c_c * (2 - c_c)
                rank_mu = np.outer(pc, pc) + delta_h * np.outer(pc, pc)
                rank_1 = np.outer(y_vec, y_vec)
                C = (1 + c_sigma * (c_cov_pos - c_sigma) + c_cov_neg * (mu_eff - 2)
                     - c_sigma * c_cov_neg) * C \
                    + c_cov_pos * rank_1 + c_cov_neg * (np.sum(w) - 1) * rank_mu
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Sigma adaptation
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 100.0)

                # Refresh C^{-1/2}
                D, B = np.linalg.eigh(C)
                D = np.maximum(D, 1e-12)
                C_inv_sqrt = B @ np.diag(1.0 / np.sqrt(D)) @ B.T

                x_mean = x_mean_new

                # Stagnation handling: re-seed from best with a wider, isotropic search
                if no_improve >= stagnation_threshold and sigma > 0.05:
                    x_mean = best_x.copy()
                    sigma = np.clip(max(sigma * 2.0, 0.5), 0.1, 50.0)
                    C = np.eye(n)
                    D, B = np.linalg.eigh(C)
                    D = np.maximum(D, 1e-12)
                    C_inv_sqrt = B @ np.diag(1.0 / np.sqrt(D)) @ B.T
                    pc = np.zeros(n)
                    ps = np.zeros(n)
                    no_improve = 0

            return best_x, best_fx, calls_used

        def lbfgsb_polish(x0, max_calls):
            """Refine from x0 with L-BFGS-B, using <= max_calls evals."""
            nonlocal best_x, best_fx, calls_used
            if max_calls < 3:
                return

            nm_calls = [0]

            def lbfgsb_func(x):
                nm_calls[0] += 1
                if nm_calls[0] > max_calls:
                    raise RuntimeError("budget exceeded")
                return func(np.clip(x, -5, 5))

            try:
                res = minimize(lbfgsb_func, x0, method='L-BFGS-B',
                               bounds=[(-5.0, 5.0)] * n,
                               options={'maxiter': max_calls, 'maxfun': max_calls,
                                        'gtol': 1e-9, 'ftol': 1e-12})
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass
            calls_used += nm_calls[0]

        # Phase 1: CMA-ES over the main budget
        cma_budget = int(budget * 0.75)
        cma_es_phase(np.zeros(n), 2.0, cma_budget)

        # Phase 2: L-BFGS-B polish with the remaining budget
        remaining = budget - calls_used
        if remaining > 5 and time.time() < deadline - 5:
            lbfgsb_polish(best_x.copy(), remaining)

        return best_x, best_fx
