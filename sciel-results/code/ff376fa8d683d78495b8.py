import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for Schaffers F7 (ill-conditioned BBOB f18)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 320.0
        n = self.dim
        B = self.budget

        def time_ok():
            return (time.time() - start) < deadline

        best_x = np.zeros(n)
        best_f = float('inf')
        calls = 0

        def ev(x):
            nonlocal calls, best_x, best_f
            if calls >= B or not time_ok():
                return float('inf')
            calls += 1
            y = np.clip(x, -5.0, 5.0)
            f = func(y)
            if f < best_f:
                best_f = f
                best_x = y.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Brief coarse sampling to find a reasonable starting region
        coarse = [np.zeros(n)]
        for _ in range(30):
            coarse.append(rng.uniform(-4.0, 4.0, size=n))
        for pt in coarse:
            if calls >= B or not time_ok():
                break
            ev(pt)

        # Phase 2: CMA-ES
        # Population size
        lambda_ = max(4, int(4 + 3 * np.log(n)))
        mu = lambda_ // 2

        # CMA-ES parameters
        sigma = 1.0
        C = np.eye(n)
        C_sqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        c_sigma = (mu + 2) / (n + 4 + mu)
        d_sigma = 1 + 2 * max(0, (mu - 1) / (n + mu - 1)) + c_sigma
        c_c = (4 + mu / n) / (n + 4 + 2 * mu / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (mu - 2 + 1 / mu) / ((n + 2) ** 2 + mu))

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # Initial mean
        m = best_x.copy()

        # CMA-ES main loop
        while calls < B and time_ok():
            # Generate population
            pop = np.zeros((lambda_, n))
            for i in range(lambda_):
                z = rng.standard_normal(n)
                # Use C_sqrt approximation: we'll use Cholesky of C
                # For efficiency, recompute C_sqrt occasionally
                if i == 0:
                    try:
                        C_sqrt = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C_sqrt = np.eye(n)
                pop[i] = m + sigma * (C_sqrt @ z)

            # Evaluate
            f_vals = np.array([ev(x) for x in pop])

            # Sort by fitness
            order = np.argsort(f_vals)
            x_sorted = pop[order]
            f_sorted = f_vals[order]

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()

            # Weighted mean of top mu
            m_new = np.sum(weights[:, None] * x_sorted[:mu], axis=0)

            # Update C and sigma
            y = (m_new - m) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y

            # Update C
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // lambda_)))) \
                      < (1 + 2 / (n + 1)) * np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // lambda_)))
            if h_sigma:
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) - C) + \
                    c_mu * np.sum(weights[:, None, None] * np.outer(x_sorted[:mu] - m, x_sorted[:mu] - m), axis=0) / sigma ** 2
            else:
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) - C + 2 * C) + \
                    c_mu * np.sum(weights[:, None, None] * np.outer(x_sorted[:mu] - m, x_sorted[:mu] - m), axis=0) / sigma ** 2

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // lambda_))) - 1))

            # Clamp sigma
            sigma = np.clip(sigma, 1e-12, 10.0)

            # Ensure C is symmetric PSD
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            m = m_new

            # Termination check: if sigma is very small, break
            if sigma < 1e-10:
                break

            # If no improvement for a while, restart with perturbation
            if calls % (lambda_ * 50) == 0 and calls > lambda_ * 10:
                if np.linalg.norm(m_new - m) < 1e-8:
                    m = m + rng.standard_normal(n) * 0.5
                    m = np.clip(m, -5.0, 5.0)
                    sigma = max(sigma, 0.5)

        # Phase 3: Final local polish with simple gradient-free refinement
        if calls < B and time_ok():
            step = 0.01
            for _ in range(200):
                if calls >= B or not time_ok():
                    break
                improved = False
                for i in range(n):
                    for sign in [1, -1]:
                        if calls >= B or not time_ok():
                            break
                        x_try = best_x.copy()
                        x_try[i] += sign * step
                        x_try = np.clip(x_try, -5.0, 5.0)
                        f_try = ev(x_try)
                        if f_try < best_f:
                            best_f = f_try
                            best_x = x_try.copy()
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-8:
                        break

        return best_x, best_f
