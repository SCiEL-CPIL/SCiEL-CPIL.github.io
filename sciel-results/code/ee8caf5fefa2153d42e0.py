"""Rewritten optimizer using GP surrogate + CMA-ES hybrid for Büche-Rastrigin f4."""
import numpy as np
import time
from scipy.stats import norm


class Optimizer:
    """GP surrogate + CMA-ES hybrid for Büche-Rastrigin f4."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        total_budget = self.budget
        rng = np.random.default_rng(7)

        best_x = None
        best_f = np.inf
        calls_made = 0
        X_list = []
        Y_list = []

        def check_deadline():
            return time.time() >= deadline or calls_made >= total_budget

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            X_list.append(x.copy())
            Y_list.append(float(f))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- GP surrogate ---
        def rbf_kernel(X1, X2, length_scale=1.5, sigma_f=10.0):
            d2 = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            d2 = np.maximum(d2, 0)
            return sigma_f**2 * np.exp(-0.5 * d2 / (length_scale**2))

        def gp_predict(X_train, y_train, X_test, length_scale=1.5, sigma_f=10.0, noise=1e-6):
            """Simple GP prediction with mean normalization."""
            y_mean = np.mean(y_train)
            y_norm = y_train - y_mean
            K = rbf_kernel(X_train, X_train, length_scale, sigma_f) + noise * np.eye(len(X_train))
            Ks = rbf_kernel(X_train, X_test, length_scale, sigma_f)
            try:
                L = np.linalg.cholesky(K + 1e-8 * np.eye(len(X_train)))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                mu = y_mean + Ks.T @ alpha
                v = np.linalg.solve(L, Ks)
                var = sigma_f**2 - np.sum(v**2, axis=0)
                var = np.maximum(var, 1e-12)
                return mu, np.sqrt(var)
            except np.linalg.LinAlgError:
                # Fallback: simple interpolation
                mu = np.array([np.mean(y_train)] * len(X_test))
                var = np.array([sigma_f**2] * len(X_test))
                return mu, np.sqrt(var)

        def gp_ei(X_train, y_train, X_candidates, length_scale=1.5, sigma_f=10.0, xi=0.01):
            """Expected Improvement acquisition."""
            y_best = np.min(y_train)
            mu, sigma = gp_predict(X_train, y_train, X_candidates, length_scale, sigma_f)
            # EI for minimization: EI = (y_best - mu) * Phi((y_best - mu)/sigma) + sigma * phi((y_best - mu)/sigma)
            z = (y_best - mu) / sigma
            ei = (y_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            ei += xi * (y_best - mu)
            return ei

        # --- CMA-ES (compact) ---
        def run_cmaes(x_init, sigma_init, max_calls):
            nonlocal calls_made, best_x, best_f

            lambda_pop = 8
            mu = lambda_pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = (weights.sum()**2) / np.sum(weights**2)

            cc = (4 + mu_eff / lambda_pop) / (lambda_pop + 4 + 2 * mu_eff / lambda_pop)
            cs = (mu_eff + 2) / (lambda_pop + mu_eff + 5)
            c1 = 2 / ((lambda_pop + 2)**2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((lambda_pop + 2)**2 + mu_eff))

            x_mean = x_init.copy()
            sigma = sigma_init
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            local_best_f = evaluate(x_mean)
            local_best_x = x_mean.copy()
            no_improve_gen = 0

            while not check_deadline() and calls_made < max_calls:
                offspring = []
                for i in range(lambda_pop):
                    if check_deadline() or calls_made >= max_calls:
                        break
                    z = rng.standard_normal(dim)
                    y = B @ (D * z)
                    x_new = x_mean + sigma * y
                    x_new = np.clip(x_new, -5, 5)
                    f_new = evaluate(x_new)
                    offspring.append((x_new, f_new))

                if len(offspring) == 0:
                    break

                offspring.sort(key=lambda x: x[1])
                x_new_mean = np.sum(
                    [weights[i] * offspring[i][0] for i in range(min(mu, len(offspring)))], axis=0
                )

                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (
                    B @ (D * (B.T @ (x_new_mean - x_mean)))
                )
                hsig = np.linalg.norm(ps) < (1.4 + 2 / (dim + 1)) * np.sqrt(
                    lambda_pop + 1 - mu_eff
                )
                if hsig:
                    sigma *= np.exp(cs / dim * (np.linalg.norm(ps) / np.sqrt(lambda_pop) - 1))
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (
                    x_new_mean - x_mean
                ) / sigma
                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(min(mu, len(offspring))):
                    y_i = (offspring[i][0] - x_mean) / sigma
                    rankmu += weights[i] * np.outer(y_i, y_i)
                C = (1 - c1 - cmu) * C + c1 * rank1 + cmu * rankmu
                C = (C + C.T) / 2
                x_mean = x_new_mean

                if offspring[0][1] < local_best_f:
                    local_best_f = offspring[0][1]
                    local_best_x = offspring[0][0].copy()
                    no_improve_gen = 0
                else:
                    no_improve_gen += 1

                try:
                    D2, B = np.linalg.eigh(C)
                    D = np.sqrt(np.maximum(D2, 1e-12))
                except Exception:
                    D2, B = np.linalg.eigh(C + 1e-10 * np.eye(dim))
                    D = np.sqrt(np.maximum(D2, 1e-12))

                if no_improve_gen >= 10 and calls_made < max_calls - 10:
                    break
                if sigma < 1e-14:
                    break

            return local_best_f, local_best_x, calls_made

        # Phase 1: Initial exploration to build GP data (100 calls)
        if not check_deadline():
            n_explore = 100
            # Mix of random and center-biased
            for i in range(n_explore):
                if check_deadline():
                    break
                if i % 3 == 0:
                    x = rng.uniform(-4, 4, dim)
                elif i % 3 == 1:
                    x = rng.uniform(-2, 2, dim)
                else:
                    x = rng.uniform(-5, 5, dim)
                evaluate(x)

        # Phase 2: GP-guided sequential search (200 calls)
        if not check_deadline() and len(X_list) >= 5:
            X_arr = np.array(X_list)
            Y_arr = np.array(Y_list)
            gp_calls = 0
            max_gp_calls = 200
            while gp_calls < max_gp_calls and not check_deadline():
                # Generate candidate points
                n_candidates = 50
                candidates = []
                for _ in range(n_candidates):
                    # Mix: around best, random, around known good points
                    r = rng.random()
                    if r < 0.3 and best_x is not None:
                        c = np.clip(best_x + rng.normal(0, 0.8, dim), -5, 5)
                    elif r < 0.5 and len(X_list) > 3:
                        # Around a randomly selected known good point
                        idx = rng.integers(0, len(X_list))
                        base = np.sort(Y_arr)[:3]  # top 3
                        good_idx = np.argsort(Y_arr)[:3]
                        chosen = good_idx[rng.integers(0, min(3, len(good_idx)))]
                        c = np.clip(X_arr[chosen] + rng.normal(0, 1.0, dim), -5, 5)
                    else:
                        c = rng.uniform(-4, 4, dim)
                    candidates.append(c)
                candidates = np.array(candidates)

                # Evaluate EI
                try:
                    ei = gp_ei(X_arr, Y_arr, candidates, length_scale=1.5, sigma_f=10.0)
                    best_idx = np.argmax(ei)
                except Exception:
                    best_idx = rng.integers(0, len(candidates))

                x_next = candidates[best_idx]
                f_next = evaluate(x_next)
                X_arr = np.vstack([X_arr, x_next.reshape(1, -1)])
                Y_arr = np.append(Y_arr, f_next)
                gp_calls += 1

        # Phase 3: CMA-ES refinement around best (remaining budget)
        if not check_deadline() and best_x is not None:
            x_init = np.clip(best_x + rng.normal(0, 0.5, dim), -5, 5)
            sigma_init = 1.0
            run_cmaes(x_init, sigma_init, total_budget)

        # Phase 4: One more diverse CMA-ES if budget remains
        if not check_deadline() and best_x is not None:
            x_init = rng.uniform(-4, 4, dim)
            for _ in range(5):
                if np.linalg.norm(x_init - best_x) > 2.0:
                    break
                x_init = rng.uniform(-4, 4, dim)
            sigma_init = 2.0
            run_cmaes(x_init, sigma_init, total_budget)

        return best_x, best_f
# EVOLVE-BLOCK-END
