import numpy as np
import time
import scipy.optimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def track_call(x, f):
            nonlocal calls_used, best_x, best_f
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_ok():
            return time.time() < deadline and calls_used < budget

        rng = np.random.default_rng(42)

        # Phase 1: Sobol quasi-random exploration for better space coverage
        n_sobol = min(200, budget // 2)
        if n_sobol > 0:
            try:
                from scipy.stats import qmc
                sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
                # Need power of 2 for Sobol
                n_sobol_pow2 = 1 << int(np.ceil(np.log2(n_sobol)))
                n_sobol_pow2 = min(n_sobol_pow2, max(2, budget // 2))
                sobol_points = sampler.random(n_sobol_pow2)
                # Map [0,1]^d to [-5,5]^d
                sobol_points = sobol_points * 10.0 - 5.0
                for i in range(n_sobol_pow2):
                    if not time_ok():
                        break
                    x = sobol_points[i]
                    fx = func(x)
                    track_call(x, fx)
            except Exception:
                # Fallback to random if Sobol fails
                for i in range(n_sobol):
                    if not time_ok():
                        break
                    x = rng.uniform(-5, 5, dim)
                    fx = func(x)
                    track_call(x, fx)

        if best_x is None:
            return np.zeros(dim), np.inf

        # Phase 2: CMA-ES from the best Sobol point
        lam = 10
        mu = lam // 2
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_w = np.sum(w)

        c_sigma = (mu_w + 2) / (dim + mu_w + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_w - 1) / (dim + 1)) - 1) + c_sigma
        c_1 = (4 - mu_w + c_sigma) / ((dim + 1.3) ** 2 + mu_w + c_sigma)
        c_mu = 2 * (mu_w - 2 + 1.0 / mu_w) / ((dim + 2) ** 2 + mu_w - 2 + 1.0 / mu_w)
        c_mu = min(c_mu, 0.8)

        mean = best_x.copy()
        sigma = 1.5
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        weights = w[:mu]

        stagnation_counter = 0
        best_f_cma = best_f
        max_stagnation = 25
        max_sigma = 4.0
        min_sigma = 1e-6

        while time_ok():
            lam_current = min(lam, budget - calls_used)
            if lam_current < 2:
                break

            ys = []
            fs = []

            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            for _ in range(lam_current):
                if not time_ok():
                    break
                z = rng.standard_normal(dim)
                y = mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                track_call(y, fy)
                ys.append(y)
                fs.append(fy)

            if len(ys) < mu:
                break

            ys = np.array(ys)
            fs = np.array(fs)

            idx = np.argsort(fs)
            ys = ys[idx]
            fs = fs[idx]

            y_w = np.sum(weights[:, None] * ys[:mu], axis=0)

            pc = (1 - c_sigma) * pc + c_sigma * (y_w - mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_w) * (y_w - mean) / sigma

            delta_sig = (np.linalg.norm(ps) / chi_n - (1 - c_sigma)) / ((1 - c_sigma) ** 2)
            C = (1 + c_1 * (delta_sig - 1) + c_mu) * C \
                - c_1 * (pc[:, None] * pc[None, :] + (delta_sig - 1) * C) \
                + c_mu * np.sum(weights[:, None, None] * (ys[:mu] - mean)[:, :, None] * (ys[:mu] - mean)[:, None, :], axis=0) / sigma ** 2

            C = (C + C.T) / 2
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-10 * np.eye(dim)

            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, min_sigma, max_sigma)

            mean = y_w
            mean = np.clip(mean, -5, 5)

            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()
                if fs[0] < best_f_cma:
                    best_f_cma = fs[0]
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
            else:
                stagnation_counter += 1

            if stagnation_counter >= max_stagnation:
                mean = best_x.copy()
                sigma = min(sigma * 2.0, max_sigma)
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                stagnation_counter = 0
                best_f_cma = best_f

        # Phase 3: Nelder-Mead local polish
        remaining = budget - calls_used
        if remaining > 10 and best_x is not None:
            try:
                nm_calls = [0]
                max_nm_calls = min(remaining - 5, 50)

                def nm_func(x):
                    nm_calls[0] += 1
                    if nm_calls[0] > max_nm_calls:
                        return best_f + 1.0
                    x_clipped = np.clip(x, -5, 5)
                    fx = func(x_clipped)
                    track_call(x_clipped, fx)
                    return fx

                result = scipy.optimize.minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_nm_calls,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )

                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5).copy()
            except Exception:
                pass

        # Phase 4: Final fine-tuning
        remaining = budget - calls_used
        if remaining > 0 and best_x is not None:
            polish_sigma = 0.001
            for _ in range(min(remaining, 15)):
                if not time_ok():
                    break
                z = rng.standard_normal(dim)
                y = best_x + polish_sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                track_call(y, fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                else:
                    polish_sigma *= 0.5

        return best_x, best_f
