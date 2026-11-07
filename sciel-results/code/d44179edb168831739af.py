"""Black-box continuous optimizer for BBOB f6 (Attractive sector).
Surrogate-based (GPR with EI) global exploration followed by CMA-ES exploitation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm
import random


class Optimizer:
    """GPR surrogate + EI for global search, then CMA-ES for local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        n = dim
        start = time.time()
        deadline = start + 280.0  # safety margin before 300s kill

        def time_left():
            return deadline - time.time()

        rng = np.random.default_rng(0)
        rng_py = random.Random(0)

        # Track all evaluations
        X_all = []
        y_all = []
        x_best = None
        f_best = np.inf

        def evaluate(x):
            nonlocal x_best, f_best
            fx = func(x)
            X_all.append(x.copy())
            y_all.append(fx)
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        # ---- Phase 1: Initial random sampling ----
        n_init = min(30, budget)
        for _ in range(n_init):
            if time_left() < 10.0:
                break
            x = rng.uniform(-5.0, 5.0, n)
            evaluate(x)

        if x_best is None:
            x_best = np.zeros(n)
            f_best = func(x_best)
            X_all.append(x_best.copy())
            y_all.append(f_best)

        # ---- Phase 2: Surrogate-based optimization (GPR + EI) ----
        # We'll use a simple GP with RBF kernel implemented manually
        def rbf_kernel(X1, X2, length_scale=1.0, var=1.0):
            # X1: (n1, d), X2: (n2, d)
            d2 = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            d2 = np.maximum(d2, 0)
            return var * np.exp(-0.5 * d2 / (length_scale ** 2))

        def fit_gp(X, y, length_scale=1.0, var=1.0, noise=1e-8):
            """Fit a GP and return predictive mean and variance functions."""
            K = rbf_kernel(X, X, length_scale, var) + noise * np.eye(len(X))
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-6 * np.eye(len(X))
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    return None
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            mu = np.mean(y)

            def predict(Xs):
                Ks = rbf_kernel(Xs, X, length_scale, var)
                mean = Ks @ alpha
                v = np.linalg.solve(L, Ks.T)
                var_pred = var - np.sum(v**2, axis=1)
                var_pred = np.maximum(var_pred, 1e-12)
                return mean, var_pred

            return predict

        def expected_improvement(predict_fn, x, f_best, xi=0.01):
            """Compute EI at point x."""
            mean, var = predict_fn(x.reshape(1, -1))
            mean = mean[0]
            var = var[0]
            std = np.sqrt(var)
            if std < 1e-10:
                return 0.0
            z = (f_best - mean - xi) / std
            ei = (f_best - mean - xi) * norm.cdf(z) + std * norm.pdf(z)
            return max(ei, 0.0)

        # GPR phase: iteratively build surrogate and pick next point
        n_gpr = min(100, budget - len(X_all))
        gpr_iters = 0
        stagnant = 0

        # Adaptive length scale
        length_scale = 1.5
        var_scale = max(np.std(y_all) ** 2, 1e-6)

        while gpr_iters < n_gpr and time_left() > 15.0:
            # Fit GP
            X_arr = np.array(X_all)
            y_arr = np.array(y_all)

            # Normalize y for GP stability
            y_mean = np.mean(y_arr)
            y_std = np.std(y_arr)
            if y_std < 1e-10:
                y_std = 1.0
            y_norm = (y_arr - y_mean) / y_std

            predict_fn = fit_gp(X_arr, y_norm, length_scale=length_scale, var=1.0, noise=1e-6)
            if predict_fn is None:
                # Fallback: just do random
                x = rng.uniform(-5.0, 5.0, n)
                evaluate(x)
                gpr_iters += 1
                continue

            # Find best EI point via multi-start local optimization
            best_ei_val = -np.inf
            best_x_ei = None

            # Candidate starting points: random + best point + neighbors of best
            n_starts = 15
            starts = [rng.uniform(-5.0, 5.0, n) for _ in range(n_starts)]
            if x_best is not None:
                starts.append(x_best.copy())
                for _ in range(5):
                    starts.append(x_best + rng.normal(0, 0.5, n))

            for s in starts:
                s = np.clip(s, -5.0, 5.0)

                def ei_neg(x, s=s):
                    x_c = np.clip(x, -5.0, 5.0)
                    return -expected_improvement(predict_fn, x_c, f_best, xi=0.01)

                try:
                    res = minimize(ei_neg, s, method='Nelder-Mead',
                                   options={'maxiter': 200, 'xatol': 0.1, 'fatol': 1e-6})
                    if res.fun < best_ei_val:
                        best_ei_val = res.fun
                        best_x_ei = np.clip(res.x, -5.0, 5.0)
                except Exception:
                    pass

            if best_x_ei is None:
                best_x_ei = rng.uniform(-5.0, 5.0, n)

            # Evaluate the chosen point
            fx = evaluate(best_x_ei)
            gpr_iters += 1

            # Check for improvement
            if fx < f_best - 1e-10:
                stagnant = 0
            else:
                stagnant += 1

            # Adapt length scale: if stagnating, increase exploration
            if stagnant >= 5:
                length_scale *= 1.5
                stagnant = 0
            elif stagnant == 0:
                length_scale = max(0.5, length_scale * 0.8)

        # ---- Phase 3: CMA-ES local refinement ----
        if time_left() < 10.0:
            return x_best, f_best

        x = x_best.copy()
        f = f_best
        sigma = 1.0
        popsize = max(4, int(4 + 3 * np.log(n)))

        # CMA-ES parameters
        c_sigma = (n + 2) / (n + 10 + popsize)
        d_sigma = 1 + 2 * max(0, np.sqrt((popsize - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + n / popsize) / (n + 4 + 2 * popsize / n)
        c_1 = (2 + 2 / (n + 1.3)) / ((n + 2) * (popsize / 2 + 2))
        c_mu = (2 + np.log(popsize / 2 + 2) - 2 / (n + 1.3)) / ((n + 2) * (popsize / 2 + 2))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        evals_cma = 0

        def _chol(M):
            w, V = np.linalg.eigh(M)
            w = np.clip(w, 1e-12, None)
            B = V @ np.diag(np.sqrt(w))
            return B

        while evals_cma < budget - len(X_all) and time_left() > 5.0:
            B = _chol(C)
            ys = rng.standard_normal((popsize, n))
            pops = np.array([x + sigma * (y @ B) for y in ys])
            pops = np.clip(pops, -5.0, 5.0)

            fs = np.empty(popsize)
            for i in range(popsize):
                if time_left() < 5.0:
                    fs[i:] = np.inf
                    break
                fs[i] = func(pops[i])
                evals_cma += 1
                X_all.append(pops[i].copy())
                y_all.append(fs[i])
                if fs[i] < f:
                    f = fs[i]
                    x_best = pops[i].copy()

            order = np.argsort(fs, kind="stable")
            lam = popsize // 2
            w = np.log(lam + 0.5) - np.log(np.arange(1, lam + 1))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)
            y_w = np.sum(w[:, None] * (pops[order[:lam]] - x), axis=0)
            y_inv = (pops[order[:lam]] - x) / sigma

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ y_w)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (evals_cma + 1)) / chi_n) < 1.4 + 2 / (n + 1)
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c)) * y_w
            C = ((1 + 2 * c_1 * hsig * (2 - c_c) - c_1 - c_mu) * C
                 + c_1 * np.outer(pc, pc)
                 + c_mu * np.sum(w[:, None, None] * np.outer(y_inv, y_inv), axis=0))
            C = 0.5 * (C + C.T)

            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(sigma, 10.0)
            sigma = max(sigma, 1e-4)

            x = pops[order[0]].copy()

        return x_best, f_best
# EVOLVE-BLOCK-END
