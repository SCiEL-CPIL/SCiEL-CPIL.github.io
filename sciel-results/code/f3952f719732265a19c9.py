"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """GP surrogate-assisted optimizer with Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 285  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track global best
        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= budget or time.time() - start_time > time_limit:
                return np.inf
            x_clipped = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            calls_used += 1
            try:
                val = func(x_clipped)
                return float(val)
            except Exception:
                return np.inf

        # Store all observations for GP
        X_obs = []
        y_obs = []

        def record(x, f):
            X_obs.append(x.copy())
            y_obs.append(f)
            nonlocal best_f, best_x
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Initial diverse sampling
        num_initial = 15
        initial_points = [np.zeros(dim)]
        for i in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[i] = sign * 2.5
                initial_points.append(p)
        while len(initial_points) < num_initial:
            initial_points.append(rng.uniform(-4, 4, dim))

        for p in initial_points:
            if calls_used >= budget or time.time() - start_time > time_limit:
                break
            f = safe_call(p)
            if np.isfinite(f):
                record(p, f)

        if calls_used >= budget or time.time() - start_time > time_limit:
            return best_x, best_f

        # Phase 2: GP surrogate with Expected Improvement
        # Fit a simple GP using RBF kernel
        def fit_gp(X, y):
            """Fit a simple GP and return mean and std functions."""
            n = len(X)
            X = np.array(X)
            y = np.array(y)
            y_mean = y.mean()
            y_std = max(y.std(), 1e-8)
            y_norm = (y - y_mean) / y_std

            # RBF kernel
            def rbf(X1, X2, length_scale=1.0):
                sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
                return np.exp(-0.5 * sq_dists / (length_scale**2))

            K = rbf(X, X, 1.0) + 1e-6 * np.eye(n)
            try:
                K_inv_y = np.linalg.solve(K, y_norm)
            except np.linalg.LinAlgError:
                K_inv_y = np.linalg.lstsq(K, y_norm, rcond=None)[0]

            def gp_mean_std(X_new):
                X_new = np.array(X_new)
                if X_new.ndim == 1:
                    X_new = X_new.reshape(1, -1)
                K_star = rbf(X_new, X, 1.0)
                mean = y_mean + y_std * (K_star @ K_inv_y)
                K_ss = rbf(X_new, X_new, 1.0)
                var = np.diag(K_ss) - np.sum(K_star**2 * K_inv_y[None, :], axis=1)
                std = np.sqrt(np.maximum(var, 1e-10))
                return mean, std

            return gp_mean_std

        def expected_improvement(x, gp_mean_std, y_best, kappa=0.01):
            """Compute EI at point x."""
            mean, std = gp_mean_std(x.reshape(1, -1))
            mean, std = mean[0], std[0]
            if std < 1e-8:
                return 0.0
            z = (y_best - mean) / std
            ei = (y_best - mean) * norm.cdf(z) + std * norm.pdf(z)
            return ei

        # GP-based exploration phase
        gp_phase_calls = 25
        gp_iterations = 0
        for _ in range(gp_phase_calls):
            if calls_used >= budget or time.time() - start_time > time_limit:
                break

            # Fit GP to current observations
            if len(X_obs) < 3:
                # Not enough data, sample randomly
                x_new = rng.uniform(-4, 4, dim)
            else:
                # Maximize EI using random restarts
                best_ei = -np.inf
                best_x_ei = None
                for _ in range(200):
                    x_cand = rng.uniform(-4.5, 4.5, dim)
                    try:
                        ei = expected_improvement(x_cand, fit_gp(X_obs, y_obs), best_f)
                    except Exception:
                        ei = 0.0
                    if ei > best_ei:
                        best_ei = ei
                        best_x_ei = x_cand

                if best_x_ei is None:
                    x_new = rng.uniform(-4, 4, dim)
                else:
                    x_new = best_x_ei

            f_new = safe_call(x_new)
            if np.isfinite(f_new):
                record(x_new, f_new)
            gp_iterations += 1

        if calls_used >= budget or time.time() - start_time > time_limit:
            return best_x, best_f

        # Phase 3: Nelder-Mead local refinement from best points
        # Sort observations by fitness
        sorted_obs = sorted(zip(y_obs, X_obs), key=lambda t: t[0])

        # Select diverse top candidates
        candidates = []
        for f, x in sorted_obs:
            if not candidates:
                candidates.append((f, x))
            else:
                # Check diversity
                diverse = True
                for _, cx in candidates:
                    if np.linalg.norm(x - cx) < 0.8:
                        diverse = False
                        break
                if diverse:
                    candidates.append((f, x))
            if len(candidates) >= 4:
                break

        num_restarts = len(candidates)
        remaining_budget = budget - calls_used
        budget_per_restart = max(15, remaining_budget // max(num_restarts, 1))

        for idx, (f0, x0) in enumerate(candidates):
            if calls_used >= budget or time.time() - start_time > time_limit:
                break

            # Nelder-Mead implementation
            x_nm = x0.copy()
            f_nm = f0

            # Initial simplex
            simplex = [x_nm.copy()]
            f_simplex = [f_nm]

            for i in range(dim):
                xi = x_nm.copy()
                step = 0.2
                xi[i] += step
                fi = safe_call(xi)
                if np.isfinite(fi):
                    record(xi, fi)
                else:
                    fi = f_nm + 1.0
                simplex.append(xi)
                f_simplex.append(fi)

            # Nelder-Mead iterations
            max_nm_iter = budget_per_restart - (dim + 1)
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

            for it in range(max_nm_iter):
                if calls_used >= budget or time.time() - start_time > time_limit:
                    break

                # Sort simplex
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]

                # Update global best
                if f_simplex[0] < best_f:
                    best_f = f_simplex[0]
                    best_x = simplex[0].copy()

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = safe_call(xr)
                if np.isfinite(fr):
                    record(xr, fr)

                if fr < f_simplex[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = safe_call(xe)
                    if np.isfinite(fe):
                        record(xe, fe)
                    if fe < fr:
                        simplex[-1] = xe
                        f_simplex[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_simplex[-1] = fr
                elif fr < f_simplex[-2]:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                else:
                    # Contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    fc = safe_call(xc)
                    if np.isfinite(fc):
                        record(xc, fc)
                    if fc < f_simplex[-1]:
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_simplex[i] = safe_call(simplex[i])
                            if np.isfinite(f_simplex[i]):
                                record(simplex[i], f_simplex[i])

                # Check convergence
                if len(f_simplex) > 1:
                    spread = max(f_simplex) - min(f_simplex)
                    if spread < 1e-10:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
