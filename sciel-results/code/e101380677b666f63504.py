import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def f(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return 1e10
            if time.time() > deadline:
                return 1e10
            calls += 1
            xc = np.clip(x, -5.0, 5.0)
            val = float(func(xc))
            if val < best_f:
                best_f = val
                best_x = xc.copy()
            return val

        # ---- Phase 1: CMA-ES with adaptive sigma ----
        # CMA-ES is well-suited for rotated, ill-conditioned landscapes like f13.
        # We implement a compact version with the key features:
        # - rank-1 and rank-mu covariance matrix adaptation
        # - cumulative step-size adaptation (CSA)
        # - population-based selection
        
        def cma_es(x0, sigma0, max_iter=100, pop_size=None):
            n = dim
            if pop_size is None:
                pop_size = 4 + int(3 * np.log(n))  # standard formula
            pop_size = max(pop_size, n + 1)
            
            # Initialize
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            C_inv_sqrt = np.eye(n)  # C^{-1/2}
            
            # Step size control parameters
            mu = pop_size // 2  # number of selected points
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            # Adaptation parameters
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            
            # State vectors
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            # Initial population
            rng = np.random.default_rng(123)
            pop = []
            for i in range(pop_size):
                z = rng.standard_normal(n)
                x_i = x_mean + sigma * (C_inv_sqrt @ z)
                pop.append(x_i)
            vals = [f(x_i) for x_i in pop]
            
            # Sort by function value
            order = np.argsort(vals)
            pop = [pop[i] for i in order]
            vals = [vals[i] for i in order]
            
            for _ in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    break
                
                # Update mean
                x_new = x_mean + sigma * sum(weights[i] * (pop[i] - x_mean) for i in range(mu))
                
                # Check if we should update
                if np.linalg.norm(x_new - x_mean) < 1e-12 * (1 + np.linalg.norm(x_mean)):
                    break
                    
                # Update step size (CSA)
                h_sigma = (np.sqrt(1 - (1 - c_sigma) ** (2 * (max_iter + 1))) /
                          np.sqrt(1 - (1 - c_sigma) ** 2))
                p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                    C_inv_sqrt @ ((x_new - x_mean) / sigma))
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) / np.sqrt(n) < (1.4 + 2 / (n + 1)) * np.sqrt((mu_eff + 1) / (n + 1)):
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(n) - np.sqrt(1 - (1 - c_sigma) ** 2)))
                
                # Update covariance
                h_c = (np.sqrt(1 - (1 - c_c) ** (2 * (max_iter + 1))) /
                      np.sqrt(1 - (1 - c_c) ** 2))
                p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma
                
                delta_h = (1 - (1 - c_c) ** (max_iter + 1)) * c_c / (1 - c_c)
                C = ((1 + c_1 * (delta_h - 1) - c_1 - c_mu) * C +
                     c_1 * np.outer(p_c, p_c) +
                     c_mu * sum(weights[i] * np.outer(pop[i] - x_mean, pop[i] - x_mean) for i in range(mu)))
                
                # Re-estimate C^{-1/2} periodically
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except:
                    pass
                
                x_mean = x_new
                
                # Generate new population
                pop = []
                for i in range(pop_size):
                    z = rng.standard_normal(n)
                    x_i = x_mean + sigma * (C_inv_sqrt @ z)
                    pop.append(x_i)
                vals = [f(x_i) for x_i in pop]
                
                order = np.argsort(vals)
                pop = [pop[i] for i in order]
                vals = [vals[i] for i in order]
            
            return x_mean, vals[0] if vals else best_f

        # ---- Phase 2: Ridge-aware refinement ----
        # After CMA-ES, we have a good estimate of the minimum region.
        # For f13 (sharp ridge), we can try to identify the ridge direction
        # and do a specialized 1D search along it.
        
        def ridge_refinement(x_start, iters=50):
            # Sample around x_start to estimate local curvature
            rng = np.random.default_rng(456)
            points = [x_start.copy()]
            vals = [f(x_start)]
            
            # Sample in random directions to estimate Hessian-like structure
            for _ in range(10):
                if time.time() > deadline or calls >= budget:
                    break
                direction = rng.standard_normal(dim)
                direction /= np.linalg.norm(direction)
                for delta in [0.1, -0.1]:
                    x_new = x_start + delta * direction
                    x_new = np.clip(x_new, -5.0, 5.0)
                    points.append(x_new.copy())
                    vals.append(f(x_new))
            
            # Find the direction of steepest descent from samples
            # Use finite differences to approximate gradient at x_start
            grad = np.zeros(dim)
            eps = 0.05
            for i in range(dim):
                if time.time() > deadline or calls >= budget:
                    break
                x_plus = x_start.copy()
                x_plus[i] += eps
                x_minus = x_start.copy()
                x_minus[i] -= eps
                f_plus = f(np.clip(x_plus, -5.0, 5.0))
                f_minus = f(np.clip(x_minus, -5.0, 5.0))
                grad[i] = (f_plus - f_minus) / (2 * eps)
            
            if np.linalg.norm(grad) < 1e-10:
                return x_start, f(x_start)
            
            # Do gradient descent with line search
            x = x_start.copy()
            fx = f(x)
            lr = 0.1
            for _ in range(iters):
                if time.time() > deadline or calls >= budget:
                    break
                
                # Recompute gradient
                grad = np.zeros(dim)
                for i in range(dim):
                    if time.time() > deadline or calls >= budget:
                        break
                    x_plus = x.copy()
                    x_plus[i] += eps
                    x_minus = x.copy()
                    x_minus[i] -= eps
                    f_plus = f(np.clip(x_plus, -5.0, 5.0))
                    f_minus = f(np.clip(x_minus, -5.0, 5.0))
                    grad[i] = (f_plus - f_minus) / (2 * eps)
                
                if np.linalg.norm(grad) < 1e-8:
                    break
                
                # Line search
                step_dir = -grad / np.linalg.norm(grad)
                best_step_lr = lr
                best_step_fx = fx
                for lr_try in [lr * 0.5, lr, lr * 1.5, lr * 2]:
                    x_try = x + lr_try * step_dir
                    x_try = np.clip(x_try, -5.0, 5.0)
                    fx_try = f(x_try)
                    if fx_try < best_step_fx:
                        best_step_fx = fx_try
                        best_step_lr = lr_try
                
                x = x + best_step_lr * step_dir
                x = np.clip(x, -5.0, 5.0)
                fx = f(x)
                
                if best_step_lr < lr:
                    lr *= 0.8
                else:
                    lr *= 1.1
            
            return x, fx

        # ---- Multi-start CMA-ES ----
        rng0 = np.random.default_rng(7)
        
        # Generate diverse starting points
        candidates = []
        for i in range(6):
            if time.time() > deadline or calls >= budget:
                break
            x_try = rng0.uniform(-3.0, 3.0, dim)
            fx_try = f(x_try)
            candidates.append((fx_try, x_try))
        
        # Add the origin as a candidate
        if calls < budget and time.time() < deadline:
            candidates.append((f(np.zeros(dim)), np.zeros(dim).copy()))
        
        candidates.sort(key=lambda t: t[0])
        
        # Run CMA-ES from the best starting points
        best_cma_x = None
        best_cma_f = np.inf
        
        # Use remaining budget wisely: allocate ~60% to CMA-ES, rest to refinement
        cma_budget = int(budget * 0.6)
        
        for i in range(min(3, len(candidates))):
            if time.time() > deadline or calls >= budget:
                break
            
            x_start = candidates[i][1].copy()
            # Estimate how many CMA-ES iterations we can afford
            remaining = budget - calls
            if remaining < 20:
                break
            
            # Each CMA-ES iteration uses pop_size evaluations
            pop_size = 4 + int(3 * np.log(dim))
            max_iter = max(5, min(50, int(remaining / pop_size)))
            
            x_cma, fx_cma = cma_es(x_start, sigma0=1.5, max_iter=max_iter)
            if fx_cma < best_cma_f:
                best_cma_f = fx_cma
                best_cma_x = x_cma
        
        # Refine with ridge-aware method
        if best_cma_x is not None and time.time() < deadline and calls < budget:
            x_final, fx_final = ridge_refinement(best_cma_x, iters=30)
            
            # Final polish with small random perturbations
            rng_final = np.random.default_rng(789)
            for _ in range(20):
                if time.time() > deadline or calls >= budget:
                    break
                x_try = x_final + 0.05 * rng_final.standard_normal(dim)
                x_try = np.clip(x_try, -5.0, 5.0)
                fx_try = f(x_try)
                if fx_try < fx_final:
                    x_final = x_try
                    fx_final = fx_try
        
        return best_x, best_f
