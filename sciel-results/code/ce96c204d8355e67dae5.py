"""Combined optimizer for BBOB f19 (Griewank-Rosenbrock).

Uses a (1+1)-ES phase for initial exploration, then switches to a GP
surrogate with a simple isotropic kernel and PI-style acquisition.
Includes a wall-clock deadline guard and adaptive phase management.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from math import exp, sqrt, log
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _init_points(self, rng, n):
        """Generate diverse initial points."""
        pts = [np.zeros(self.dim)]
        for _ in range(n - 1):
            r = rng.random()
            if r < 0.25:
                pts.append(rng.uniform(-5, 5, self.dim))
            elif r < 0.50:
                pts.append(rng.integers(-3, 4, self.dim).astype(float))
            elif r < 0.75:
                pts.append(rng.uniform(-2, 2, self.dim))
            else:
                pts.append(rng.integers(-1, 2, self.dim).astype(float))
        return pts

    def _N_pdf(self, z):
        return exp(-0.5 * z * z) / sqrt(2.0 * np.pi)

    def _N_cdf(self, z):
        """Normal CDF approximation."""
        if z >= 0:
            t = 1.0 / (1.0 + 0.2316419 * z)
            poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
            return 1.0 - poly * exp(-0.5 * z * z) / sqrt(2.0 * np.pi)
        else:
            return 1.0 - self._N_cdf(-z)

    def _fit_gp_simple(self, X, y):
        """Fit a simple GP with isotropic kernel. Returns (l2, alpha, beta)."""
        n, d = X.shape
        if n == 1:
            return 1.0, np.ones(1), float(y[0])
        
        # Estimate length-scale from data spread
        d0 = X - X[0]
        d2_all = (d0 * d0).sum(axis=1)
        d2_pos = d2_all[d2_all > 0]
        if len(d2_pos) > 0:
            l = float(np.clip(np.sqrt(2.0 * np.mean(d2_pos)), 0.1, 10.0))
        else:
            l = 1.0
        l2 = l * l
        
        # Build kernel matrix
        d2 = np.zeros((n, n))
        for i in range(n):
            diff = X[i] - X
            d2[i] = (diff * diff).sum(axis=1)
        K = np.exp(-0.5 * d2 / l2)
        K[np.arange(n), np.arange(n)] += 1e-6
        
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K[np.arange(n), np.arange(n)] += 1e-4
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K[np.arange(n), np.arange(n)] += 1e-3
                L = np.linalg.cholesky(K)
        
        alpha = np.linalg.solve(L, np.ones(n))
        beta = float(np.mean(y))
        return l2, alpha, beta

    def _acq_pi(self, X, y, l2, alpha, beta, x):
        """Probability-of-improvement acquisition function."""
        n = X.shape[0]
        dx = x - X
        d2 = (dx * dx).sum(axis=1) / l2
        K = np.exp(-0.5 * d2)
        
        mu = beta + (K - 0.5 * d2) @ alpha
        v = (K - 0.5 * d2) * alpha
        var = float(v.sum())
        sd = sqrt(max(var, 1e-12))
        
        if sd < 1e-9:
            return -mu
        
        z = (beta - mu) / sd
        phi = self._N_pdf(z)
        Phi = self._N_cdf(z)
        c = phi * max(z, 0.0) + (beta - mu) * (1.0 - Phi)
        c = max(c, 0.0)
        return -mu - c

    def _acq_pi_grad(self, X, y, l2, alpha, beta, x):
        """PI acquisition with gradient."""
        n = X.shape[0]
        dx = x - X
        d2 = (dx * dx).sum(axis=1) / l2
        K = np.exp(-0.5 * d2)
        
        mu = beta + (K - 0.5 * d2) @ alpha
        v = (K - 0.5 * d2) * alpha
        var = float(v.sum())
        sd = sqrt(max(var, 1e-12))
        
        # Gradient of mu: dmu/dx = -sum_i alpha_i * K_i * dx_i / l2
        g_mu = -(K * dx) / l2
        g_mu = g_mu.sum(axis=0)
        
        if sd < 1e-9:
            val = -mu
            return val, -g_mu
        
        z = (beta - mu) / sd
        phi = self._N_pdf(z)
        Phi = self._N_cdf(z)
        c = phi * max(z, 0.0) + (beta - mu) * (1.0 - Phi)
        c = max(c, 0.0)
        val = -mu - c
        
        # Gradient of acquisition: dval/dx = -g_mu - (c/sd) * g_mu
        g_acq = -g_mu - (c / sd) * g_mu
        return val, g_acq

    def _optimize_acq(self, X, y, l2, alpha, beta, rng, best_x):
        """Optimize acquisition function with multi-start L-BFGS-B."""
        best_val = -np.inf
        best_xq = None
        
        # Multi-start candidates
        cands = [np.zeros(self.dim)]
        cands.append(np.clip(best_x, -5, 5))
        for _ in range(8):
            cands.append(rng.uniform(-5, 5, self.dim))
        
        for x0 in cands:
            x0 = np.clip(x0, -5, 5)
            try:
                res = minimize(
                    self._acq_pi_grad,
                    x0,
                    args=(X, y, l2, alpha, beta),
                    method="L-BFGS-B",
                    bounds=[(-5, 5)] * self.dim,
                    jac=True,
                    options={"maxiter": 20, "maxls": 15},
                )
                xv = np.clip(res.x, -5, 5)
                val, _ = self._acq_pi_grad(X, y, l2, alpha, beta, xv)
                if val > best_val:
                    best_val = val
                    best_xq = xv
            except Exception:
                continue
        
        if best_xq is None:
            best_xq = cands[0]
        return best_xq

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        rng = np.random.default_rng(0)
        
        # Phase 1: Diverse initial sampling
        init_pts = self._init_points(rng, 12)
        xs, ys = [], []
        best = (None, np.inf)
        
        def sample(x):
            nonlocal best
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = float(func(x))
            xs.append(x)
            ys.append(f)
            if f < best[1]:
                best = (x.copy(), f)
            return f
        
        for x in init_pts:
            if time.time() >= deadline:
                break
            sample(x)
        
        # Phase 2: (1+1)-ES to refine local search
        x = best[0].copy()
        fx = best[1]
        sigma = 1.5
        n_es = 25
        for _ in range(n_es):
            if time.time() >= deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = sample(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1
            else:
                sigma *= 0.85
        
        # Phase 3: GP model-and-optimize
        n_min_gp = 10
        gp_count = 0
        for _ in range(self.budget - len(xs)):
            if time.time() >= deadline:
                break
            
            X = np.array(xs)
            y = np.array(ys, dtype=float)
            n = len(xs)
            
            if n < n_min_gp:
                # Not enough points: explore around best
                xq = np.clip(best[0] + 1.0 * rng.standard_normal(self.dim), -5, 5)
            else:
                # Try GP
                try:
                    l2, alpha, beta = self._fit_gp_simple(X, y)
                    xq = self._optimize_acq(X, y, l2, alpha, beta, rng, best[0])
                    gp_count += 1
                except Exception:
                    # Fall back to local exploration
                    xq = np.clip(best[0] + 0.8 * rng.standard_normal(self.dim), -5, 5)
            
            sample(xq)
        
        return best
# EVOLVE-BLOCK-END
