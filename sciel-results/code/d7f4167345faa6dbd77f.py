"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    CMA-ES optimizer with PCA-aligned covariance initialization and
    L-BFGS-B polish for BBOB f18 (Schaffers F7 ill-conditioned).
    
    Strategy:
      - Random sampling phase to seed the portfolio.
      - Initialize CMA-ES covariance from top samples (PCA) to align
        with the funnel structure of the ill-conditioned landscape.
      - Full CMA-ES with covariance matrix adaptation.
      - Final L-BFGS-B polish from the global best for tight local refinement.
      - Explicit wall-clock deadline guard (safety margin before 330 s).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 320.0  # safety margin vs 330 s wall-clock
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(12345)

        calls = 0

        def safe_call(x):
            nonlocal calls
            calls += 1
            return func(x)

        # ---- Phase 1: random sampling to seed the portfolio ----
        n_samples = min(60, budget // 8)
        samples = []
        for _ in range(n_samples):
            if calls >= budget or time.time() >= deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            samples.append((fx, x.copy()))

        if not samples:
            return 0.0

        samples.sort(key=lambda t: t[0])
        best_fx, best_x = samples[0][0], samples[0][1]

        # ---- Initialize covariance from top samples (PCA) ----
        n_top = max(5, min(15, len(samples) // 3))
        top_samples = np.array([s[1] for s in samples[:n_top]])

        if n_top > 1:
            cov = np.cov(top_samples, rowvar=False)
            if cov.size == 0:
                cov = np.eye(dim)
            # Regularize
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-6)
            cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
        else:
            cov = np.eye(dim)

        # ---- Phase 2: CMA-ES ----
        remaining = budget - calls
        if remaining <= 0 or time.time() >= deadline:
            return best_fx

        # Reserve budget for polish phase
        polish_budget = max(30, int(remaining * 0.15))
        cma_budget = remaining - polish_budget

        pop_size = max(6, int(4 + 3 * np.log(dim)))
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)

        c_sigma = (pop_size + 2) / (dim + 4)
        c1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        cmu = min(0.5, c1 * (pop_size // 2 - 1) / 2)
        d_sigma = 1 + 2 * max(0, (dim / pop_size - 1) / 3) + cmu
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        m = best_x.copy()
        sigma = 1.0
        C = cov.copy()
        try:
            C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
        except np.linalg.LinAlgError:
            C = np.eye(dim)
            C_sqrt = np.eye(dim)

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        stall_count = 0
        max_stall = 10
        best_cma_fx = best_fx

        while calls < cma_budget + n_samples:
            if time.time() > deadline - 5.0:
                break

            # Generate population
            offspring = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                y = m + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)

            # Evaluate offspring
            evaluated = []
            for y in offspring:
                if calls >= cma_budget + n_samples or time.time() > deadline - 5.0:
                    break
                fy = safe_call(y)
                evaluated.append((fy, y))

            if len(evaluated) < pop_size:
                break

            evaluated.sort(key=lambda t: t[0])

            # Track best
            if evaluated[0][0] < best_fx:
                best_fx = evaluated[0][0]
                best_x = evaluated[0][1].copy()
                stall_count = 0
            else:
                stall_count += 1

            # Update mean
            w = weights
            m_new = np.sum(w[:, np.newaxis] * np.array([e[1] for e in evaluated[:mu]]), axis=0)
            
            # Update covariance and sigma
            diff = np.array([e[1] for e in evaluated[:mu]]) - m
            C = (1 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) - C) + cmu * np.sum(w[:, np.newaxis] * np.outer(diff, diff), axis=0)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update path accumulators
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt.T @ (m_new - m) / sigma)
            p_c = (1 - c1) * p_c + np.sqrt(c1 * (2 - c1) * mu_eff) * (m_new - m) / sigma
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            m = m_new
            
            # Recompute Cholesky
            try:
                C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                # Regularize
                C = C + 1e-6 * np.eye(dim)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    C_sqrt = np.eye(dim)
                    sigma = 1.0
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)

            # Stall detection: restart with larger sigma
            if stall_count >= max_stall:
                sigma *= 1.5
                stall_count = 0
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)

        # ---- Phase 3: L-BFGS-B polish ----
        if calls < budget and time.time() < deadline:
            x_start = best_x.copy()

            polish_calls = [0]
            def polish_func(x):
                polish_calls[0] += 1
                return func(np.clip(x, -5, 5))

            try:
                max_iter = min(50, (budget - calls) // 5)
                if max_iter >= 1:
                    res = minimize(polish_func, x_start, method='L-BFGS-B',
                                   bounds=[(-5, 5)] * dim,
                                   options={'maxiter': max_iter, 'maxfun': budget - calls - 1})
                    if res.fun < best_fx:
                        best_fx = res.fun
                        best_x = res.x
            except Exception:
                pass

        return best_fx
# EVOLVE-BLOCK-END
