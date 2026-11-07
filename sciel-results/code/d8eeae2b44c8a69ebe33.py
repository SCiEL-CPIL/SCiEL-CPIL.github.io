"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with time guard for Rosenbrock rotated."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, rng, x0, f0, sigma0, max_iter, deadline):
        """Robust CMA-ES for dim=5."""
        n = self.dim
        lam = 10  # population size
        mu = lam // 2  # number of parents
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / (w ** 2).sum()
        
        # Parameters
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        
        sigma = sigma0
        x = x0.copy()
        f_best = f0
        x_best = x0.copy()
        
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        for _ in range(max_iter):
            if time.time() > deadline - 3:
                break
            
            # Sample
            Z = rng.standard_normal((lam, n))
            L = np.linalg.cholesky(C + 1e-12 * np.eye(n))
            X = x + sigma * (Z @ L.T)
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(X[i]) for i in range(lam)])
            
            # Sort
            idx = np.argsort(F)
            Xs = X[idx]
            Fs = F[idx]
            
            if Fs[0] < f_best:
                f_best = Fs[0]
                x_best = Xs[0].copy()
            
            # New mean
            x_new = np.sum(w[:, None] * Xs[:mu], axis=0)
            
            # Evolution paths
            y = (x_new - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (L.T @ y)
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * y
            
            # Sigma update
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Covariance update
            delta = (1 + 2 * max(0, (np.linalg.norm(pc)**2 / n - 1) / (1 - c_c)))
            delta = min(delta, 1 / (1 - c_c))
            C = (1 - c_1 - c_mu + delta * c_1) * C \
                + c_1 * (np.outer(pc, pc) + (delta - 1) * np.eye(n)) \
                + c_mu * np.sum(w[:mu, None] * (Xs[:mu] - x)[..., None] * (Xs[:mu] - x)[..., None], axis=0) / sigma**2
            
            C = (C + C.T) / 2
            x = x_new
        
        return x_best, f_best

    def __call__(self, func):
        start = time.time()
        deadline = start + 280  # 50s safety margin
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)
        
        # Phase 1: Random sampling (10% of budget)
        n_rand = max(5, budget // 10)
        x_best = None
        f_best = np.inf
        
        for _ in range(n_rand):
            if time.time() > deadline - 60:
                break
            x = rng.uniform(-5, 5, dim)
            f = func(x)
            if f < f_best:
                f_best = f
                x_best = x.copy()
        
        if x_best is None:
            x_best = np.zeros(dim)
            f_best = func(x_best)
        
        # Phase 2: CMA-ES from best point
        if time.time() < deadline - 60:
            x_best, f_best = self._cma_es(
                func, rng, x_best, f_best, 2.0,
                min(budget - n_rand, 200), deadline
            )
        
        # Phase 3: Restart from different points if time allows
        if time.time() < deadline - 30:
            for _ in range(2):
                if time.time() > deadline - 15:
                    break
                xr = rng.uniform(-5, 5, dim)
                fr = func(xr)
                if fr < f_best:
                    f_best = fr
                    x_best = xr.copy()
                x_best, f_best = self._cma_es(
                    func, rng, x_best, f_best, 1.5,
                    min(50, budget // 5), deadline
                )
        
        return x_best, f_best
# EVOLVE-BLOCK-END
