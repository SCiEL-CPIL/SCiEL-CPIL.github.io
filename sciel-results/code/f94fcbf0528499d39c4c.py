"""Black-box continuous optimizer for BBOB f4 (Büche-Rastrigin) using GP surrogate + EI."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """GP surrogate-based optimizer with Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        rng = np.random.default_rng(42)
        dim = self.dim

        # State
        X = []  # list of evaluated points
        Y = []  # list of function values
        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), self.lb, self.ub)
            f = float(func(x))
            X.append(x.copy())
            Y.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initial random sampling to build GP (100 evaluations)
        n_init = 100
        for i in range(n_init):
            x = rng.uniform(self.lb, self.ub, dim)
            evaluate(x)

        # GP hyperparameters
        # We'll use a simple GP with Matern 5/2 kernel
        # For efficiency, we'll use a sparse approximation or just work with all points
        # Since dim=5 and up to 500 points, direct GP is feasible but let's be careful

        # Normalize inputs for GP stability
        x_min = np.array([self.lb] * dim)
        x_max = np.array([self.ub] * dim)

        def normalize(x):
            if len(x.shape) == 1:
                return (x - x_min) / (x_max - x_min)
            return (x - x_min) / (x_max - x_min)

        def denormalize(x):
            if len(x.shape) == 1:
                return x * (x_max - x_min) + x_min
            return x * (x_max - x_min) + x_min

        def matern52_kernel(K, length_scale, sigma_f):
            """Matern 5/2 kernel."""
            d = np.sqrt(np.sum((K[:, None, :] - K[None, :, :]) ** 2, axis=-1))
            sqrt3_d = np.sqrt(3) * d / length_scale
            cov = (1 + sqrt3_d + (sqrt3_d ** 2) / 3) * np.exp(-sqrt3_d)
            return cov * (sigma_f ** 2)

        def fit_gp_hyperparams(Xn, Yn, max_iter=50):
            """Fit GP hyperparameters using simple grid/random search."""
            n = len(Yn)
            if n < 5:
                return 1.0, 1.0, 0.01

            # Standardize Y for stability
            y_mean = np.mean(Yn)
            y_std = np.std(Yn) + 1e-10
            Yn_std = (Yn - y_mean) / y_std

            best_lik = -np.inf
            best_ls = 1.0
            best_sf = 1.0
            best_noise = 0.01

            # Try a few hyperparameter combinations
            ls_candidates = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
            sf_candidates = [0.5, 1.0, 1.5, 2.0, 3.0]
            noise_candidates = [0.001, 0.01, 0.05, 0.1]

            for ls in ls_candidates:
                for sf in sf_candidates:
                    for noise in noise_candidates:
                        try:
                            K = matern52_kernel(Xn, ls, sf)
                            K += noise * np.eye(n)
                            # Compute log marginal likelihood
                            L = np.linalg.cholesky(K)
                            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Yn_std))
                            log_det = 2 * np.sum(np.log(np.diag(L)))
                            lik = -0.5 * (Yn_std @ alpha + log_det + n * np.log(2 * np.pi))
                            if lik > best_lik:
                                best_lik = lik
                                best_ls = ls
                                best_sf = sf
                                best_noise = noise
                        except np.linalg.LinAlgError:
                            continue

            return best_ls, best_sf, best_noise

        def gp_predict(Xn_new, Xn_train, Yn_train, length_scale, sigma_f, noise):
            """Predict mean and std at Xn_new using GP trained on Xn_train, Yn_train."""
            n = len(Yn_train)
            K_train = matern52_kernel(Xn_train, length_scale, sigma_f)
            K_train += noise * np.eye(n)
            K_cross = matern52_kernel(np.vstack([Xn_new, Xn_train]), length_scale, sigma_f)
            K_new = K_cross[:len(Xn_new), n:]
            K_new_new = matern52_kernel(Xn_new, length_scale, sigma_f)

            try:
                L = np.linalg.cholesky(K_train)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Yn_train))
                v = np.linalg.solve(L, K_new.T)
                mean = K_new @ alpha
                var = np.diag(K_new_new) - np.sum(v ** 2, axis=0)
                var = np.maximum(var, 1e-10)
                std = np.sqrt(var)
            except np.linalg.LinAlgError:
                # Fallback: use nearest neighbor interpolation
                mean = np.zeros(len(Xn_new))
                std = np.full(len(Xn_new), np.std(Yn_train) + 1e-10)
                for i, xn in enumerate(Xn_new):
                    dists = np.sum((Xn_train - xn) ** 2, axis=1)
                    idx = np.argmin(dists)
                    mean[i] = Yn_train[idx]

            return mean, std

        # Phase 2: Sequential EI-based optimization (remaining budget)
        n_remaining = self.budget - len(X)

        # Precompute GP hyperparameters initially
        Xn = np.array([normalize(x) for x in X])
        Yn = np.array(Y)
        y_mean = np.mean(Yn)
        y_std = np.std(Yn) + 1e-10
        Yn_std = (Yn - y_mean) / y_std

        length_scale, sigma_f, noise = fit_gp_hyperparams(Xn, Yn_std)

        for step in range(n_remaining):
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Fit GP hyperparameters periodically (every 20 steps)
            if step % 20 == 0 and len(X) >= 10:
                Xn = np.array([normalize(x) for x in X])
                Yn = np.array(Y)
                y_mean = np.mean(Yn)
                y_std = np.std(Yn) + 1e-10
                Yn_std = (Yn - y_mean) / y_std
                length_scale, sigma_f, noise = fit_gp_hyperparams(Xn, Yn_std)
            else:
                Xn = np.array([normalize(x) for x in X])
                Yn = np.array(Y)
                y_mean = np.mean(Yn)
                y_std = np.std(Yn) + 1e-10
                Yn_std = (Yn - y_mean) / y_std

            # Generate candidate points for EI maximization
            # Mix of: random, near-best, and diverse
            n_candidates = 200
            candidates = []

            # 50% random
            n_rand = n_candidates // 2
            candidates.extend([rng.uniform(0, 1, dim) for _ in range(n_rand)])

            # 30% near current best (in normalized space)
            n_near = n_candidates // 3
            best_idx = np.argmin(Y)
            best_xn = normalize(best_x)
            for _ in range(n_near):
                perturbation = rng.normal(0, 0.1, dim)
                cand = np.clip(best_xn + perturbation, 0, 1)
                candidates.append(cand)

            # 20% diverse (far from existing points)
            n_div = n_candidates - n_rand - n_near
            for _ in range(n_div):
                for attempt in range(10):
                    cand = rng.uniform(0, 1, dim)
                    # Check distance to existing points
                    if len(Xn) > 0:
                        dists = np.sum((Xn - cand) ** 2, axis=1)
                        if np.min(dists) > 0.01:
                            break
                candidates.append(cand)

            candidates = np.array(candidates)

            # Predict GP at candidates
            mean, std = gp_predict(candidates, Xn, Yn_std, length_scale, sigma_f, noise)

            # Expected Improvement
            best_y_so_far = np.min(Yn_std)
            improvement = best_y_so_far - mean
            z = improvement / (std + 1e-10)
            from scipy.stats import norm
            EI = improvement * norm.cdf(z) + std * norm.pdf(z)

            # Select best candidate
            idx = np.argmax(EI)
            x_new_norm = candidates[idx]
            x_new = denormalize(x_new_norm)
            evaluate(x_new)

        # Final local refinement around best point
        if time.time() < deadline - 10.0 and best_x is not None:
            # Use scipy L-BFGS-B for a few steps
            def wrapper(x):
                return float(func(np.clip(x, self.lb, self.ub)))

            try:
                res = minimize(wrapper, best_x, method='L-BFGS-B',
                               bounds=[(self.lb, self.ub)] * dim,
                               options={'maxiter': 20, 'maxfun': 30})
                if res.fun < best_f:
                    best_x = np.clip(res.x, self.lb, self.ub)
                    best_f = res.fun
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
