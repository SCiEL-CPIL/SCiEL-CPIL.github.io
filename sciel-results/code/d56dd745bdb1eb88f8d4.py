import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 320.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)

        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=np.float64).flatten(), -5, 5)
            if x.shape != (dim,):
                return np.inf
            try:
                fx = func(x)
                calls_made += 1
                if np.isfinite(fx) and fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                return np.inf

        # 1. Initialization: Sample a few points to get a good starting mean
        # Use 4 points to balance exploration vs. budget for CMA-ES
        n_init = 4
        init_points = rng.uniform(-5, 5, (n_init, dim))
        init_fvals = np.array([safe_eval(p) for p in init_points])
        
        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # Find the best initial point
        best_init_idx = np.argmin(init_fvals)
        mean = init_points[best_init_idx].copy()
        
        # 2. CMA-ES Phase
        # Allocate ~60% of budget to CMA-ES to leave more for local refinement
        cma_budget_target = int(budget * 0.60)
        
        lam = 12  # Population size
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = (2 * (mu_eff - 2 + 2 / mu_eff)) / ((dim + 2) ** 2 + mu_eff)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Initial state
        sigma = 2.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        def sample_from_cov(mean, sigma, C, rng):
            val, vec = np.linalg.eigh(C)
            val = np.clip(val, 1e-12, None)
            sqrt_val = np.sqrt(val)
            z = rng.standard_normal(dim)
            return mean + sigma * (vec @ (z * sqrt_val))
        
        while calls_made < cma_budget_target and time.time() < deadline:
            # Generate population
            pop = []
            fvals = []
            
            for _ in range(lam):
                if calls_made >= cma_budget_target or time.time() > deadline:
                    break
                x_new = sample_from_cov(mean, sigma, C, rng)
                x_new = np.clip(x_new, -5, 5)
                fx = safe_eval(x_new)
                pop.append(x_new)
                fvals.append(fx)
            
            if len(pop) < lam:
                break
                
            fvals = np.array(fvals)
            order = np.argsort(fvals)
            sorted_pop = np.array([pop[i] for i in order])
            
            # Update mean
            new_mean = np.sum(weights[:, np.newaxis] * sorted_pop[:mu], axis=0)
            
            # Update evolution paths
            val, vec = np.linalg.eigh(C)
            val = np.clip(val, 1e-12, None)
            inv_sqrt_C = vec @ np.diag(1.0 / np.sqrt(val)) @ vec.T
            
            diff = (new_mean - mean) / sigma
            pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * (inv_sqrt_C @ diff)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ diff)
            
            # Update sigma
            norm_ps = np.linalg.norm(ps)
            delta_sigma = (norm_ps / chi_n - 1) * d_sigma
            sigma *= np.exp(c_sigma * delta_sigma)
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update C
            y = (sorted_pop[:mu] - mean) / sigma
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(y[i], y[i])
            
            decay = 1 - c_1 - c_mu
            rank_one = np.outer(pc, pc) + (delta_sigma * c_sigma * (2 - c_sigma)) * C
            C = decay * C + c_1 * rank_one + c_mu * rank_mu
            C = (C + C.T) / 2
            
            mean = new_mean
            
            # Stop if convergence is reached or sigma is tiny
            if sigma < 1e-8:
                break

        # 3. Local Refinement Phase
        # Use the remaining budget for a local search around the best point found.
        remaining_budget = budget - calls_made
        if remaining_budget > 10 and time.time() < deadline - 5:
            try:
                x0_local = best_x.copy()
                
                # Use Nelder-Mead for robust derivative-free local optimization
                result = minimize(
                    safe_eval, 
                    x0_local, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': int(remaining_budget / 2), 
                        'maxfev': remaining_budget,
                        'xatol': 1e-8,
                        'fatol': 1e-12
                    }
                )
            except Exception:
                pass
        
        return best_x, best_f
