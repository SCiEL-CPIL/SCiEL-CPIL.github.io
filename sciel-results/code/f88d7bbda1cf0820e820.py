import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = 328.0  # Safety margin for 330s limit
        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf

        rng = np.random.default_rng(42)

        # 1. Initialization: Sample 10 points with better coverage
        init_budget = 10
        candidates = []
        # Include origin and random points spread across the box
        candidates.append(np.zeros(dim))
        # Add points at various scales along each axis
        for d in range(dim):
            p = np.zeros(dim)
            p[d] = 2.0
            candidates.append(p)
        # Add random points to fill remaining budget
        while len(candidates) < init_budget:
            candidates.append(rng.uniform(-3.0, 3.0, dim))
        
        cand_vals = []
        for c in candidates:
            c_clipped = np.clip(c, -5, 5)
            val = func(c_clipped)
            cand_vals.append(val)
            
        best_idx = int(np.argmin(cand_vals))
        x = np.clip(candidates[best_idx], -5, 5).copy()
        fx = cand_vals[best_idx]
        used = init_budget

        if fx < best_f:
            best_f = fx
            best_x = x.copy()

        # 2. CMA-ES
        # Parameters for dim=5
        lam = 8  # Smaller population for more iterations and earlier progress
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        sigma = 0.5  # Start smaller to avoid overshooting
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Cholesky factor for sampling
        L = np.linalg.cholesky(C)

        polish_budget = 40
        cma_budget = max(0, total_budget - used - polish_budget)
        max_iters = cma_budget // lam

        for it in range(max_iters):
            if time.time() > deadline:
                break
            if used >= total_budget:
                break

            # Sample lambda points using Cholesky
            Y = np.zeros((lam, dim))
            for i in range(lam):
                z = rng.standard_normal(dim)
                Y[i] = L @ z

            # Evaluate
            X = np.clip(x[None, :] + sigma * Y, -5, 5)
            F = np.array([func(xi) for xi in X])
            used += lam

            # Update best
            for i in range(lam):
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()

            # Sort
            order = np.argsort(F)
            F_sorted = F[order]
            Y_sorted = Y[order]
            X_sorted = X[order]

            # Mean update
            x_old = x.copy()
            x = x_old + sigma * np.sum(w[:, None] * Y_sorted[:mu], axis=0)
            x = np.clip(x, -5, 5)

            # Covariance matrix adaptation
            ys = Y_sorted[:mu]
            mean_y = np.sum(w[:, None] * ys, axis=0)

            # Update p_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * mean_y

            # Update C
            H = np.zeros((dim, dim))
            for i in range(mu):
                H += w[i] * np.outer(ys[i], ys[i])

            coeff_c = (1 - c_c) * (1 - c_sigma)
            coeff_pc = c_c * (1 - c_c) * d_sigma
            coeff_h = c_sigma * (1 - 1.0 / mu_eff)

            C_new = coeff_c * C + coeff_pc * np.outer(p_c, p_c) + coeff_h * H
            C_new = (C_new + C_new.T) / 2

            # Ensure positive definite for Cholesky
            eigvals, eigvecs = np.linalg.eigh(C_new)
            if np.min(eigvals) < 1e-12:
                eigvals = np.clip(eigvals, 1e-12, None)
                C_new = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            C = C_new
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to identity if Cholesky fails
                C = np.eye(dim)
                L = np.eye(dim)

            # Step-size control
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * mean_y
            sigma *= np.exp((1.0 / (5 * dim + 5)) * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

        # 3. Polish with Nelder-Mead (two stages: coarse then fine)
        if time.time() < deadline and used < total_budget:
            # Use the best point found so far
            x_start = best_x.copy()
            
            # Define a wrapper for scipy that respects the budget and time
            calls_made = 0
            max_calls = total_budget - used
            
            def scipy_func(x):
                nonlocal calls_made
                if calls_made >= max_calls:
                    return best_f + 1.0 # Penalty
                if time.time() > deadline:
                    return best_f + 1.0
                x_clipped = np.clip(x, -5, 5)
                val = func(x_clipped)
                calls_made += 1
                if val < best_f:
                    best_f = val
                    best_x = x_clipped.copy()
                return val

            # Stage 1: Coarse polish
            try:
                res1 = minimize(scipy_func, x_start, method='Nelder-Mead', 
                               options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
                if res1.fun < best_f:
                    best_f = res1.fun
                    best_x = np.clip(res1.x, -5, 5)
            except Exception:
                pass

            # Stage 2: Fine polish if budget remains
            if time.time() < deadline and used < total_budget:
                try:
                    res2 = minimize(scipy_func, best_x.copy(), method='Nelder-Mead', 
                                   options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14})
                    if res2.fun < best_f:
                        best_f = res2.fun
                        best_x = np.clip(res2.x, -5, 5)
                except Exception:
                    pass

        return best_x, best_f
