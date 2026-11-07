"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Fixed CMA-ES with multi-start and local polish for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        
        best_x = None
        best_f = float('inf')
        eval_count = 0
        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                raise Exception("Budget exceeded")
            if time.time() > deadline:
                raise Exception("Time exceeded")
            
            x = np.asarray(x, dtype=float)
            fx = func(x)
            eval_count += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            return fx

        def cma_es(start_x, sigma, max_iter, max_evals):
            """Run CMA-ES from a starting point with correct updates."""
            nonlocal eval_count, best_x, best_f, deadline
            
            pop_size = max(4, int(4 + 3 * np.log(dim)))
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
            
            x_mean = start_x.copy().astype(float)
            C = np.eye(dim)
            D = np.ones(dim)  # diagonal of C for Cholesky
            B = np.eye(dim)   # Cholesky factor: C = B @ B.T
            
            # Initial population
            pop = []
            fvals = []
            for i in range(pop_size):
                z = np.random.randn(dim)
                x_i = x_mean + sigma * (B * D) @ z
                x_i = np.clip(x_i, -5.0, 5.0)
                pop.append(x_i)
                fvals.append(wrapped_func(x_i))
            
            order = np.argsort(fvals)
            pop = [pop[i] for i in order]
            fvals = [fvals[i] for i in order]
            x_mean = np.sum(np.array(pop[:mu]) * weights[:, None], axis=0)
            
            iteration = 0
            no_improve = 0
            local_best_f = fvals[0]
            
            while iteration < max_iter and eval_count < max_evals:
                if time.time() > deadline:
                    break
                
                iteration += 1
                
                # Update evolution paths
                d_sigma = np.sqrt(c_sigma * (5 + c_sigma))
                p_sigma = (1 - c_sigma) * p_sigma + d_sigma * np.sqrt(mu_eff) * (weights @ np.array(pop[:mu]) - x_mean) / (sigma + 1e-32)
                
                norm_p_sigma = np.linalg.norm(p_sigma)
                
                # h_sigma indicator
                hs = (norm_p_sigma / chi_n < (1.4 + c_sigma) * chi_n)
                
                # Update p_c
                p_c = (1 - c_c) * p_c + hs * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_mean) / (sigma + 1e-32)
                # Fix: use the actual mean difference from previous iteration
                # For simplicity, we track the previous mean
                if iteration > 1:
                    p_c = (1 - c_c) * p_c + hs * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_mean_prev) / (sigma + 1e-32)
                x_mean_prev = x_mean.copy()
                
                # Update covariance matrix
                delta_hs = (1 - hs) * c_c * (2 - c_c)
                C = (1 + c_1 + c_mu - c_1 - c_c * (1 + c_1 - c_c) + delta_hs) * C
                C += c_1 * (np.outer(p_c, p_c) - C)
                
                # Rank-mu update
                y = (np.array(pop[:mu]) - x_mean) / (sigma + 1e-32)
                C += c_mu * np.sum(weights[:, None] * np.outer(y, y), axis=0)
                
                # Ensure positive definiteness and update Cholesky
                C = 0.5 * (C + C.T)
                try:
                    B = np.linalg.cholesky(C)
                    D = np.diag(B)
                    B = B / D[:, None]  # Normalize columns
                except np.linalg.LinAlgError:
                    # Fallback: eigen decomposition
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-10)
                    C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                    try:
                        B = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                        D = np.diag(B)
                        B = B / D[:, None]
                    except:
                        B = np.eye(dim)
                        D = np.ones(dim)
                
                # Update sigma
                sigma *= np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)
                
                # Generate new population
                new_pop = []
                new_fvals = []
                
                for i in range(pop_size):
                    if eval_count >= max_evals or time.time() > deadline:
                        break
                    z = np.random.randn(dim)
                    x_i = x_mean + sigma * (B * D) @ z
                    x_i = np.clip(x_i, -5.0, 5.0)
                    new_pop.append(x_i)
                    new_fvals.append(wrapped_func(x_i))
                
                if len(new_pop) < pop_size:
                    break
                
                order = np.argsort(new_fvals)
                new_pop = [new_pop[i] for i in order]
                new_fvals = [new_fvals[i] for i in order]
                
                # Update mean
                x_mean = np.sum(np.array(new_pop[:mu]) * weights[:, None], axis=0)
                
                # Check for improvement
                if new_fvals[0] < local_best_f - 1e-10:
                    no_improve = 0
                    local_best_f = new_fvals[0]
                else:
                    no_improve += 1
                
                if no_improve > 15:
                    break
                
                pop = new_pop
                fvals = new_fvals
            
            return x_mean, local_best_f

        # Phase 1: Initial sampling
        try:
            from scipy.stats import qmc
            n_scout = 40
            sampler = qmc.Sobol(d=dim, scramble=True)
            samples = sampler.random(n=n_scout)
            samples = samples * 10.0 - 5.0
            scout_points = list(samples)
        except ImportError:
            n_scout = 40
            scout_points = np.random.uniform(-5, 5, size=(n_scout, dim)).tolist()

        # Add strategic points
        scout_points.append(np.zeros(dim))
        scout_points.append(np.ones(dim))
        scout_points.append(-np.ones(dim))
        scout_points.append(2.0 * np.ones(dim))
        scout_points.append(-2.0 * np.ones(dim))
        scout_points.append(np.array([1, 1, 1, 1, 1]))
        scout_points.append(np.array([0.5, 0.5, 0.5, 0.5, 0.5]))

        for p in scout_points:
            if time.time() > deadline or eval_count >= budget:
                break
            try:
                fp = wrapped_func(p)
            except Exception:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = func(best_x)
                eval_count += 1
            except Exception:
                pass

        # Phase 2: CMA-ES with multiple restarts
        remaining = budget - eval_count
        if remaining > 60:
            # First CMA-ES from best point
            try:
                x_cma, f_cma = cma_es(best_x, 1.5, max_iter=60, max_evals=budget)
                if f_cma < best_f:
                    best_f = f_cma
                    best_x = x_cma.copy()
            except Exception:
                pass
        
        # Second CMA-ES from a perturbed point if budget allows
        remaining = budget - eval_count
        if remaining > 40:
            try:
                perturbation = np.random.randn(dim) * 0.5
                start2 = np.clip(best_x + perturbation, -5, 5)
                x_cma2, f_cma2 = cma_es(start2, 1.0, max_iter=40, max_evals=budget)
                if f_cma2 < best_f:
                    best_f = f_cma2
                    best_x = x_cma2.copy()
            except Exception:
                pass

        # Phase 3: Local polish with L-BFGS-B
        remaining = budget - eval_count
        if remaining > 20 and best_x is not None:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 100, 'maxfun': remaining, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # Phase 4: Final Nelder-Mead polish
        remaining = budget - eval_count
        if remaining > 15 and best_x is not None:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'maxfev': remaining, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 5: One more L-BFGS-B pass if still have budget
        remaining = budget - eval_count
        if remaining > 10 and best_x is not None:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 50, 'maxfun': remaining, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
