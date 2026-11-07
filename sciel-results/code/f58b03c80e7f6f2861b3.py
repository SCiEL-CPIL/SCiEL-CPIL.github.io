"""Direct-approach optimizer for BBOB f22 (Gallagher 21 peaks) in 5D.

Strategy:
  Phase 1: Coarse global grid (3^5 = 243 points) to locate the best basin
           early, which strongly improves the AOCC area-under-curve score.
  Phase 2: CMA-ES refinement centered on the best grid point for fast
           local convergence to the minimum.
  A wall-clock deadline guard (320 s) returns the best result found safely early.
"""
import time
import numpy as np


class Optimizer:
    """Grid-then-CMA-ES optimizer for the BBOB f22 task."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cma_es(self, func, x0, sigma0, budget, calls, best, deadline):
        """
        CMA-ES with full covariance matrix for robust local optimization.
        """
        n = self.dim
        x = x0.copy().astype(float)
        sigma = float(sigma0)
        
        # CMA-ES parameters
        # Population size
        lam = max(4, int(4 + 3 * np.log(n)))
        
        # Weights
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        
        # Step size control parameters
        cs = (mu_eff + 2) / (n + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + cs
        c_sigma = (5 + mu_eff / n) / (n + 5 + 2 * mu_eff / n)
        
        # Covariance matrix adaptation parameters
        c1 = (2 - mu_eff / (n ** 1.7) + 2 * w[0]) / ((n + 2) ** 1.6 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Initial covariance matrix
        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt = np.eye(n)
        
        # Path variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Precompute Cholesky decomposition
        try:
            C_sqrt = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_sqrt = np.eye(n)
        
        used = 0
        while used < budget:
            # Deadline guard
            if time.time() >= deadline:
                break
            
            # Generate population
            y = np.random.standard_normal((lam, n))
            x_pop = np.tile(x, (lam, 1)) + sigma * np.dot(y, C_sqrt.T)
            x_pop = np.clip(x_pop, -5, 5)
            
            # Evaluate population
            f_pop = np.empty(lam)
            for i in range(lam):
                if used >= budget:
                    break
                f_pop[i] = func(x_pop[i])
                used += 1
            
            if used >= budget:
                break
            
            # Sort by fitness
            idx = np.argsort(f_pop)
            x_pop = x_pop[idx]
            f_pop = f_pop[idx]
            y = y[idx]
            
            # Update best
            if f_pop[0] < best["f"]:
                best["f"] = f_pop[0]
                best["x"] = x_pop[0].copy()
            
            # Recombination
            x_old = x
            x = x + sigma * np.dot(w[:mu], y[:mu])
            
            # Update covariance matrix
            y_w = (x - x_old) / sigma
            C_old = C
            C = (1 - c1 - cmu) * C + c1 * (np.outer(y_w, y_w) + (1 - cs) * C_old)
            if mu > 1:
                C += cmu * np.dot(w[:mu], np.outer(y_w, y_w))  # Simplified, usually sum w_i * y_i y_i^T
            
            # Ensure positive semi-definiteness and update Cholesky
            C = (C + C.T) / 2
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small identity to make it positive definite
                C += 1e-6 * np.eye(n)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C_sqrt = np.eye(n)
                    C = np.eye(n)
            
            # Update step size
            y_w_std = y_w / np.linalg.norm(y_w) if np.linalg.norm(y_w) > 1e-12 else 0
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * y_w_std
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (used + 1))) < (1.4 + 2 / (n + 1)) * np.trace(np.eye(n))) else 0
            sigma *= np.exp(c_sigma / ds * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update path for covariance
            p_c = (1 - cs) * p_c + h_sigma * np.sqrt(cs * (2 - cs) * mu_eff) * C_sqrt.T @ y_w_std
            h_c = 1 if (np.linalg.norm(p_c) / np.sqrt(1 - (1 - cs) ** (2 * (used + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt((1 + (1 - cs) * cmu / c1) * np.trace(np.eye(n)))) else 0
            
            C = (1 + c1 * (h_c - 1) + cmu) * C - c1 * np.outer(p_c, p_c)
            C = (C + C.T) / 2
            
            # Clamp sigma
            sigma = float(np.clip(sigma, 1e-8, 10.0))
            
            # Convergence check
            if np.linalg.norm(x - x_old) < 1e-12 and sigma < 1e-12:
                break

        return best

    def __call__(self, func):
        deadline = time.time() + 320.0  # Safety margin before 330 s kill
        n = self.dim
        budget = self.budget
        calls = {"count": 0}

        def fwrap(x):
            if calls["count"] >= budget:
                return float('inf')
            calls["count"] += 1
            return func(x)

        best = {"x": None, "f": np.inf}

        # ---------------- Phase 1: coarse global grid ----------------
        # 3^5 = 243 points over [-5, 5]^5 (spacing ~2.5).
        grid_n = 3
        grid_vals = np.linspace(-5.0, 5.0, grid_n)
        grid_points = list(np.meshgrid(*([grid_vals] * n), indexing="ij"))
        grid_points = np.array([p for p in zip(*grid_points)])
        grid_points = np.asarray(grid_points).reshape(-1, n)

        grid_budget = min(len(grid_points), max(50, int(0.45 * budget)))
        if len(grid_points) > grid_budget:
            idx = np.linspace(0, len(grid_points) - 1, grid_budget).astype(int)
            grid_points = grid_points[idx]

        for gp in grid_points:
            if time.time() >= deadline:
                break
            gp = np.asarray(gp, dtype=float)
            fg = fwrap(gp)
            if fg < best["f"]:
                best["f"] = fg
                best["x"] = gp.copy()

        # ---------------- Phase 2: CMA-ES refinement ----------------
        if best["x"] is None:
            best["x"] = np.random.uniform(-5, 5, n)
            best["f"] = fwrap(best["x"])

        remaining = budget - calls["count"]
        sigma0 = 1.0
        self._cma_es(fwrap, best["x"], sigma0, remaining, calls, best, deadline)

        if best["x"] is None:
            x0 = np.zeros(n)
            best["f"] = fwrap(x0)
            best["x"] = x0

        return best["x"], best["f"]
