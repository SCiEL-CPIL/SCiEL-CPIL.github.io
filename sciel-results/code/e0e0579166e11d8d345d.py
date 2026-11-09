"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with Nelder-Mead + BFGS polish for BBOB f19 (Griewank-Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Track all evaluations
        eval_count = [0]

        def safe_call(x):
            if eval_count[0] >= budget:
                return best_f + 1e10
            if time.time() > deadline:
                return best_f + 1e10
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            try:
                fx = func(x)
                eval_count[0] += 1
                if fx < best_f:
                    best_f = fx
                    best_x[:] = x
                return float(fx)
            except:
                return best_f + 1e10

        # Initial center
        best_x = np.zeros(n)
        best_f = float('inf')
        fx0 = safe_call(best_x)
        best_f = fx0

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(n))  # ~11 for n=5
        mu = lam // 2
        weights = np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size
        sigma = 2.0

        # Covariance matrix
        C = np.eye(n)
        Cs = np.zeros(n)  # evolution path for covariance

        # Evolution path for step size
        ps = np.zeros(n)
        cs = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        ds = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + cs

        # Eigen decomposition cache
        eigvals, eigvecs = np.linalg.eigh(C)

        # CMA-ES main loop - use ~70% of budget for global search
        cma_budget = int(budget * 0.70)

        while eval_count[0] < cma_budget and time.time() < deadline:
            # Update eigen decomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
            except:
                eigvals = np.ones(n) * 1e-10
                eigvecs = np.eye(n)

            # Generate population
            Y = np.zeros((lam, n))
            for i in range(lam):
                z = rng.standard_normal(n)
                y = eigvecs @ (np.sqrt(eigvals) * z)
                Y[i] = np.clip(best_x + sigma * y, -5, 5)

            # Evaluate population
            F = np.array([safe_call(Y[i]) for i in range(lam)])

            # Update best
            idx_best = np.argmin(F)
            if F[idx_best] < best_f:
                best_f = F[idx_best]
                best_x = Y[idx_best].copy()

            # Sort by fitness
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]

            # Weighted mean of best mu individuals
            m = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)

            # Evolution path for step size
            inv_sigma = 1.0 / sigma if sigma > 1e-30 else 1.0
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * inv_sigma * (eigvecs @ (np.sqrt(eigvals) * (eigvecs.T @ (m - best_x))))

            # Update sigma
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Update mean (use m as new center)
            old_x = best_x.copy()
            best_x = m.copy()

            # Evolution path for covariance
            Cs = (1 - cs) * Cs + np.sqrt(cs * (2 - cs) * mu_eff) * inv_sigma * (eigvecs @ (np.sqrt(eigvals) * (eigvecs.T @ (m - old_x))))

            # Update covariance matrix
            delta_sigma = (1 + 2 * max(0, (np.linalg.norm(ps) / chi_n - 1)) * cs) * ds
            C = (1 - cs * (2 - cs) - cs * delta_sigma * mu_eff) * C \
                + cs * delta_sigma * np.outer(Cs, Cs) \
                + mu_eff * np.sum(weights[:, None, None] * np.outer(Y_sorted[:mu] - old_x, Y_sorted[:mu] - old_x), axis=0)

            # Ensure positive definiteness
            eigvals_new, _ = np.linalg.eigh(C)
            if np.any(eigvals_new < 1e-20):
                C = C + 1e-10 * np.eye(n)

        # Polish phase: Nelder-Mead + BFGS on remaining budget
        polish_budget = budget - eval_count[0]
        if polish_budget > 10 and time.time() < deadline:
            # Nelder-Mead local refinement
            nm_options = {'maxiter': min(polish_budget // 3, 200), 'xatol': 1e-8, 'fatol': 1e-10}
            try:
                res_nm = minimize(safe_call, best_x.copy(), method='Nelder-Mead', options=nm_options)
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = res_nm.x.copy()
            except:
                pass

            # BFGS refinement (using numerical gradient via finite differences)
            if time.time() < deadline and eval_count[0] < budget - 5:
                def bfgs_func(x):
                    return safe_call(x)
                
                # Use a tighter Nelder-Mead result as starting point for BFGS
                try:
                    res_bfgs = minimize(bfgs_func, best_x.copy(), method='BFGS', 
                                        options={'maxiter': min(polish_budget // 4, 100), 'gtol': 1e-6})
                    if res_bfgs.fun < best_f:
                        best_f = res_bfgs.fun
                        best_x = res_bfgs.x.copy()
                except:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
