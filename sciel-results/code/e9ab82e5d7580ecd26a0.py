"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset.

Strategy:
  1. CMA-ES global search with adaptive step size for robust convergence.
  2. Nelder-Mead local polish to exploit the smooth, unimodal geometry of f14.
  3. Final short CMA-ES refinement with small sigma if budget remains.

The hidden affine transform (shift + rotation + offset) preserves the unimodal,
smooth nature of f14, so local methods like NM are effective once the global
search has placed us near the basin of attraction.
"""
import time
import numpy as np


class Optimizer:
    """CMA-ES + Nelder-Mead polish optimizer for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 310.0  # safety margin before 330s kill
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget
        calls = 0

        def elapsed():
            return time.time() - start

        def still_ok():
            return elapsed() < deadline

        def make_call(x):
            nonlocal calls
            if calls >= budget:
                return None
            fx = float(func(x))
            calls += 1
            return fx

        # ---- Phase 0: Initial evaluation at origin ----
        x0 = np.zeros(dim)
        fx0 = make_call(x0)
        if fx0 is None:
            return x0, np.inf

        best_x = x0.copy()
        best_f = fx0

        # ---- Phase 1: CMA-ES global search ----
        def run_cmaes(x_init, f_init, sigma_init, max_calls):
            nonlocal calls, best_x, best_f
            n = dim
            # CMA-ES parameters
            lambda_ = 4 * int(1 + 3 * n)  # population size
            mu = lambda_ // 2  # number of parents
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w**2)
            # Step size control
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            # Mean update
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
            # Covariance matrix
            C = np.eye(n)
            # Evolution path
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            # Initial sigma
            sigma = sigma_init
            # Initial mean
            x_mean = x_init.copy()
            f_mean = f_init
            # Initial population
            y = np.zeros((lambda_, n))
            x_pop = np.zeros((lambda_, n))
            f_pop = np.zeros(lambda_)
            
            for gen in range(max_calls // lambda_ + 1):
                if not still_ok() or calls >= max_calls:
                    break
                
                # Sample population
                for i in range(lambda_):
                    y[i] = rng.standard_normal(n)
                    x_pop[i] = x_mean + sigma * y[i]
                    x_pop[i] = np.clip(x_pop[i], -5.0, 5.0)
                    f_pop[i] = make_call(x_pop[i])
                    if f_pop[i] is None:
                        return x_mean, f_mean, sigma
                    if f_pop[i] < best_f:
                        best_f = f_pop[i]
                        best_x = x_pop[i].copy()
                
                # Sort by fitness
                order = np.argsort(f_pop)
                y = y[order]
                x_pop = x_pop[order]
                f_pop = f_pop[order]
                
                # Update mean
                x_old = x_mean.copy()
                x_mean = np.sum(w[:, np.newaxis] * x_pop[:mu], axis=0)
                f_mean = f_pop[0]
                
                # Update evolution paths
                # Compute inverse sqrt of C
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.sqrt(eigvals)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
                
                y_weighted = np.sum(w[:, np.newaxis] * y[:mu], axis=0)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_inv_sqrt @ y_weighted
                chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
                if np.linalg.norm(p_sigma) < 1.4 * (1 + c_sigma) * chi_n:
                    c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (x_mean - x_old) / sigma
                else:
                    c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (x_mean - x_old) / sigma
                
                # Update covariance matrix
                delta_sigma = (np.linalg.norm(p_sigma) / chi_n - 1) * d_sigma * c_sigma
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    rank_mu += w[i] * np.outer(y[i], y[i])
                C = (1 + c_sigma * delta_sigma - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                # Symmetrize
                C = (C + C.T) / 2
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-9, 10.0)
                
                # Check convergence
                if np.max(f_pop[:mu]) - np.min(f_pop[:mu]) < 1e-12:
                    break
                
                if calls >= max_calls:
                    break
            
            return x_mean, f_mean, sigma

        # Allocate ~40% of budget to global CMA-ES, rest to local polish
        cmaes_budget = max(50, int(budget * 0.40))
        
        # First CMA-ES run from current best
        x, f, sigma = run_cmaes(best_x, best_f, 2.0, calls + cmaes_budget)

        # ---- Phase 2: Nelder-Mead local polish ----
        if still_ok() and calls < budget:
            x_best = best_x.copy()
            f_best = best_f

            # Simplex: best_x + small perturbations along each axis
            simplex_size = 0.5
            simplex = [x_best.copy()]
            for i in range(dim):
                vertex = x_best.copy()
                vertex[i] += simplex_size
                simplex.append(np.clip(vertex, -5.0, 5.0))

            # Evaluate simplex vertices
            simplex_f = []
            for v in simplex:
                fv = make_call(v)
                if fv is None:
                    break
                simplex_f.append(fv)
                if fv < f_best:
                    f_best = fv
                    x_best = v.copy()
                    best_f = fv
                    best_x = v.copy()
            else:
                # All simplex vertices evaluated; run Nelder-Mead
                simplex_f = np.array(simplex_f)
                simplex = np.array(simplex)

                # Standard Nelder-Mead parameters
                alpha = 1.0   # reflection
                gamma = 2.0   # expansion
                rho = 0.5     # contraction
                sigma_nm = 0.5  # shrink

                nm_max_iter = 200
                nm_iter = 0

                while (nm_iter < nm_max_iter and still_ok() and calls < budget):
                    nm_iter += 1
                    # Sort by function value
                    order = np.argsort(simplex_f)
                    simplex = simplex[order]
                    simplex_f = simplex_f[order]

                    # Check convergence
                    if np.max(simplex_f) - np.min(simplex_f) < 1e-10:
                        break

                    x_worst = simplex[-1]
                    f_worst = simplex_f[-1]
                    x_best_nm = simplex[0]
                    f_best_nm = simplex_f[0]

                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    x_ref = centroid + alpha * (centroid - x_worst)
                    x_ref = np.clip(x_ref, -5.0, 5.0)
                    f_ref = make_call(x_ref)
                    if f_ref is None:
                        break

                    if f_ref < f_best_nm:
                        # Expansion
                        x_exp = centroid + gamma * (x_ref - centroid)
                        x_exp = np.clip(x_exp, -5.0, 5.0)
                        f_exp = make_call(x_exp)
                        if f_exp is None:
                            break
                        if f_exp < f_ref:
                            x_new, f_new = x_exp, f_exp
                        else:
                            x_new, f_new = x_ref, f_ref
                        simplex[-1] = x_new
                        simplex_f[-1] = f_new
                    elif f_ref < simplex_f[-2]:
                        # Accept reflection
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                    else:
                        # Contraction
                        x_con = centroid + rho * (x_worst - centroid)
                        x_con = np.clip(x_con, -5.0, 5.0)
                        f_con = make_call(x_con)
                        if f_con is None:
                            break
                        if f_con < f_worst:
                            simplex[-1] = x_con
                            simplex_f[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = x_best_nm + sigma_nm * (simplex[i] - x_best_nm)
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                f_sh = make_call(simplex[i])
                                if f_sh is None:
                                    break
                                simplex_f[i] = f_sh
                            else:
                                pass
                            break

                    # Update best
                    for i in range(dim + 1):
                        if simplex_f[i] < f_best:
                            f_best = simplex_f[i]
                            x_best = simplex[i].copy()
                            best_f = f_best
                            best_x = x_best.copy()

        # ---- Phase 3: Final short CMA-ES refinement if budget remains ----
        if still_ok() and calls < budget:
            x, f, sigma = run_cmaes(best_x, best_f, 0.1, budget)

        return best_x, float(best_f)
