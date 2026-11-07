"""Model-and-optimize black-box optimizer for BBOB f18 (Schaffers F7).

The "model" is a multivariate Gaussian (mean m, covariance C); its parameters
are fitted/optimized against the objective each generation by re-weighting the
best samples (CMA-ES). Adaptive covariance tracks the ill-conditioned basin
faster than the (1+1)-ES seed, improving early AOCC.
"""
import time
import numpy as np


class Optimizer:
    """CMA-ES (adaptive-Gaussian model) with box repair and a wall-clock guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        WALL = 280.0                      # evaluator kills at 330 s; keep margin
        calls = [0]

        def time_ok():
            return (time.time() - t0) < WALL

        def safe(x):
            """One recorded, exception-safe, budget/time-guarded evaluation."""
            if calls[0] >= self.budget or not time_ok():
                return None
            try:
                v = float(func(np.asarray(x, dtype=float)))
            except Exception:
                v = 1e18
            if not np.isfinite(v):
                v = 1e18
            calls[0] += 1
            return v

        n = self.dim
        rng = np.random.default_rng(0)

        # ---- CMA-ES hyper-parameters
        lam = 20
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / float(w @ w)
        c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
        c1 = 2.0 / (n ** 1.6 + 18.0)
        cmu = 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff)
        chiN = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

        best_x = np.zeros(n)
        best_f = np.inf
        stagnation_limit = 15 # Generations without improvement before restart

        def evaluate(pop):
            fs = []
            for x in pop:
                v = safe(x)
                if v is None:
                    fs.append(1e18)
                else:
                    fs.append(v)
            return np.asarray(fs)

        # Initialize CMA-ES state
        m = rng.uniform(-2.0, 2.0, n)
        sigma = 2.5
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        g = 0
        last_best = np.inf
        g_since_improve = 0

        # Run CMA-ES until budget/time limit
        while calls[0] < self.budget and time_ok():
            if self.budget - calls[0] < lam:
                break

            Z = rng.standard_normal((lam, n))
            Y = (Z @ B.T) * D
            X = m[None, :] + sigma * Y
            X = np.clip(X, -5.0, 5.0)
            f = evaluate(X)

            fi = int(np.argmin(f))
            if f[fi] < best_f:
                best_f = float(f[fi])
                best_x = X[fi].copy()

            idx = np.argsort(f)
            top = idx[:mu]
            Ys = Y[top] / sigma
            Xs = X[top]

            ps = (1.0 - c_sigma) * ps + np.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * (w @ Ys)
            hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - c_sigma) ** (2 * (g + 1)))) / chiN < \
                   1.4 + 2.0 / (n + 1)
            pc = (1.0 - c_c) * pc + hsig * np.sqrt(c_c * (2.0 - c_c) * mu_eff) * (w @ (Xs - m[None, :])) / sigma

            C = (1.0 - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) + (hsig - 1.0 + c1) * C)
            C += cmu * sum(wi * np.outer(y, y) for wi, y in zip(w, Ys))

            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chiN - 1.0))
            sigma = float(np.clip(sigma, 1e-8, 10.0))

            m = w @ Xs
            m = np.clip(m, -5.0, 5.0)

            try:
                eig, B = np.linalg.eigh(C)
                eig = np.maximum(eig, 1e-12)
                D = np.sqrt(eig)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                B = np.eye(n)
                D = np.ones(n)

            g += 1

            # Check for stagnation
            if best_f < last_best:
                last_best = best_f
                g_since_improve = 0
            else:
                g_since_improve += 1

            if g_since_improve > stagnation_limit:
                # Restart: center on best, increase sigma, reset C
                m = best_x.copy()
                sigma = 2.0
                C = np.eye(n)
                B = np.eye(n)
                D = np.ones(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                g = 0
                last_best = best_f
                g_since_improve = 0

        # ---- spend any remaining budget with a cheap (1+1)-ES tail ----
        # Use best_x as center for tail
        if calls[0] < self.budget and time_ok():
            m_tail = best_x
            sigma_tail = 0.5
            while calls[0] < self.budget and time_ok():
                y = m_tail + (sigma_tail * 0.5) * rng.standard_normal(n)
                y = np.clip(y, -5.0, 5.0)
                fy = safe(y)
                if fy is None:
                    break
                if fy < best_f:
                    best_f, best_x = fy, y.copy()
                    m_tail = best_x
                    sigma_tail *= 1.2
                else:
                    sigma_tail *= 0.85
                    sigma_tail = float(np.clip(sigma_tail, 1e-8, 10.0))

        return best_x, float(best_f)
