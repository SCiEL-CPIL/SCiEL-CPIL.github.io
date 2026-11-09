"""Surrogate-assisted (GP) + Nelder-Mead hybrid optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer for BBOB f11 (Discus):
    - Phase 1: structured + random exploration to seed the surrogate.
    - Phase 2: Gaussian-Process surrogate-assisted acquisition (Expected Improvement).
    - Phase 3: Nelder-Mead local polish on the best point found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 310  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        X = []
        Y = []
        evals = 0

        def _eval(x):
            nonlocal evals
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = func(x)
            X.append(x.copy())
            Y.append(float(f))
            evals += 1
            return f

        # --- Phase 1: Initial exploration ---
        # Center
        _eval(np.zeros(dim))

        # Axis probes: both directions, several magnitudes
        for i in range(dim):
            for d in [1.0, 2.0, 3.0, 4.0]:
                for sign in [1.0, -1.0]:
                    if evals >= budget:
                        break
                    p = np.zeros(dim)
                    p[i] = sign * d
                    _eval(p)
                if evals >= budget:
                    break
            if evals >= budget:
                break

        # Random points for diversity
        n_rand = min(25, budget - evals)
        for _ in range(n_rand):
            if evals >= budget:
                break
            _eval(rng.uniform(-4.5, 4.5, dim))

        if evals >= budget:
            arr = np.array(X)
            yarr = np.array(Y)
            bi = int(np.argmin(yarr))
            return arr[bi], yarr[bi]

        # --- Phase 2: GP surrogate-assisted search ---
        # Reserve budget for Nelder-Mead polish
        nm_budget = max(40, int(budget * 0.15))
        gp_end = budget - nm_budget

        # GP hyperparameters
        # RBF kernel: k(x,y) = sigma_f^2 * exp(-0.5 * ||x-y||^2 / l^2)
        sigma_f = 1.0
        length_scale = 1.5
        noise = 1e-6

        def _rbf_kernel(X1, X2, ls, sf):
            """Compute RBF kernel between two sets of points."""
            # X1: (n1, d), X2: (n2, d)
            d2 = np.sum(X1 ** 2, axis=1)[:, None] - 2.0 * X1 @ X2.T + np.sum(X2 ** 2, axis=1)[None, :]
            d2 = np.maximum(d2, 0.0)
            return sf ** 2 * np.exp(-0.5 * d2 / (ls ** 2))

        def _gp_predict(X_train, Y_train, X_test, ls, sf, noise):
            """Return mean and std at X_test given training data."""
            n = len(X_train)
            K = _rbf_kernel(X_train, X_train, ls, sf)
            K[np.diag_indices(n)] += noise
            m = np.mean(Y_train)
            y = Y_train - m
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-6 * np.eye(n)
                L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            K_s = _rbf_kernel(X_test, X_train, ls, sf)
            mu = m + K_s @ alpha
            v = np.linalg.solve(L, K_s.T)
            var = sf ** 2 - np.sum(v ** 2, axis=1)
            var = np.maximum(var, 1e-10)
            std = np.sqrt(var)
            return mu, std

        def _expected_improvement(mu, std, best_f, xi=0.01):
            """EI acquisition with exploration parameter xi."""
            improvement = best_f - mu - xi
            eps = 1e-10
            std_safe = np.maximum(std, eps)
            z = improvement / std_safe
            # Standard normal CDF and PDF
            from scipy.stats import norm
            ei = improvement * norm.cdf(z) + std_safe * norm.pdf(z)
            ei[std < eps] = 0.0
            return ei

        from scipy.stats import norm

        # Adaptive GP: occasionally re-fit length scale and signal variance
        # by maximizing a simple log-likelihood on the current data.
        def _fit_gp_hyperparams(X_train, Y_train, rng_local, n_iter=15):
            """Simple random-search fit of (log_ls, log_sf)."""
            best_ll = -np.inf
            best_ls, best_sf = 1.5, 1.0
            n = len(X_train)
            K_base = _rbf_kernel(X_train, X_train, 1.0, 1.0)  # shape matrix
            for _ in range(n_iter):
                ls = np.exp(rng_local.uniform(np.log(0.3), np.log(5.0)))
                sf = np.exp(rng_local.uniform(np.log(0.1), np.log(10.0)))
                K = _rbf_kernel(X_train, X_train, ls, sf)
                K[np.diag_indices(n)] += noise
                m = np.mean(Y_train)
                y = Y_train - m
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    continue
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                ll = -0.5 * y @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
                if ll > best_ll:
                    best_ll = ll
                    best_ls, best_sf = ls, sf
            return best_ls, best_sf

        X_arr = np.array(X)
        Y_arr = np.array(Y)

        # Initial hyperparameter fit
        ls, sf = _fit_gp_hyperparams(X_arr, Y_arr, rng)

        gp_iter = 0
        stagnation = 0
        best_f_gp = float(np.min(Y_arr))
        best_x_gp = X_arr[int(np.argmin(Y_arr))].copy()
        max_stagnation = 20

        while evals < gp_end:
            if time.time() > deadline:
                break

            gp_iter += 1

            # Refit hyperparameters every 15 iterations or at start
            if gp_iter % 15 == 1:
                X_arr = np.array(X)
                Y_arr = np.array(Y)
                ls, sf = _fit_gp_hyperparams(X_arr, Y_arr, rng)

            # Generate candidate points
            # Mix: random candidates + local jitter around current best
            n_cand = 60
            candidates = []

            # Random candidates
            n_rand_cand = int(n_cand * 0.4)
            for _ in range(n_rand_cand):
                candidates.append(rng.uniform(-5.0, 5.0, dim))

            # Jitter around best
            n_jitter = int(n_cand * 0.35)
            for _ in range(n_jitter):
                candidates.append(np.clip(best_x_gp + rng.normal(0, 0.5, dim), -5, 5))

            # Latin-hypercube-ish spread
            n_lh = n_cand - n_rand_cand - n_jitter
            for _ in range(n_lh):
                p = rng.permutation(dim)
                for i in range(dim):
                    p[i] = rng.uniform(-5.0, 5.0)
                candidates.append(p)

            cand_arr = np.array(candidates)

            # Predict with GP
            mu, std = _gp_predict(X_arr, Y_arr, cand_arr, ls, sf, noise)

            # EI acquisition
            ei = _expected_improvement(mu, std, best_f_gp, xi=0.005)

            # Pick top candidates by EI, evaluate a few of them
            top_k = min(5, len(candidates))
            top_idx = np.argsort(ei)[::-1][:top_k]

            improved_this_iter = False
            for idx in top_idx:
                if evals >= gp_end:
                    break
                x_new = cand_arr[idx]
                f_new = _eval(x_new)
                if f_new < best_f_gp:
                    best_f_gp = f_new
                    best_x_gp = x_new.copy()
                    improved_this_iter = True

            if improved_this_iter:
                stagnation = 0
            else:
                stagnation += 1

            X_arr = np.array(X)
            Y_arr = np.array(Y)

            if stagnation >= max_stagnation:
                # Perturb: add a few random points to escape local optimum
                for _ in range(min(5, gp_end - evals)):
                    if evals >= gp_end:
                        break
                    p = rng.uniform(-5.0, 5.0, dim)
                    f_p = _eval(p)
                    if f_p < best_f_gp:
                        best_f_gp = f_p
                        best_x_gp = p.copy()
                        stagnation = 0
                # Re-fit hyperparameters
                X_arr = np.array(X)
                Y_arr = np.array(Y)
                ls, sf = _fit_gp_hyperparams(X_arr, Y_arr, rng)
                stagnation = 0

        # --- Phase 3: Nelder-Mead local polish ---
        x_nm = best_x_gp.copy()
        f_nm = best_f_gp

        # Initialize simplex
        simplex = [x_nm.copy()]
        simplex_f = [f_nm]
        step = 0.5
        for i in range(dim):
            v = x_nm.copy()
            v[i] += step
            simplex.append(v)
            simplex_f.append(_eval(v))
            if evals >= budget:
                break

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        max_nm_iter = 200
        nm_iter = 0

        while evals < budget and nm_iter < max_nm_iter:
            if time.time() > deadline:
                break

            nm_iter += 1

            # Sort simplex
            order = np.argsort(simplex_f)
            simplex = [simplex[j] for j in order]
            simplex_f = [simplex_f[j] for j in order]

            # Update global best
            if simplex_f[0] < f_nm:
                f_nm = simplex_f[0]
                x_nm = simplex[0].copy()

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]

            # Reflection
            x_r = centroid + alpha * (centroid - worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = _eval(x_r)

            if f_r < simplex_f[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - worst)
                x_e = np.clip(x_e, -5, 5)
                if evals >= budget:
                    break
                f_e = _eval(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                x_c = centroid + rho * (worst - centroid)
                x_c = np.clip(x_c, -5, 5)
                if evals >= budget:
                    break
                f_c = _eval(x_c)
                if f_c < simplex_f[-1]:
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        if evals >= budget:
                            break
                        simplex_f[i] = _eval(simplex[i])
                    if evals >= budget:
                        break

        # Final best
        all_x = np.array(X)
        all_y = np.array(Y)
        bi = int(np.argmin(all_y))
        best_x_final = all_x[bi]
        best_f_final = all_y[bi]

        return best_x_final, best_f_final
# EVOLVE-BLOCK-END
