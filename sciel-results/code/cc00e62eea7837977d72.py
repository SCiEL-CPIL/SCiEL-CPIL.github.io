"""Rewritten optimizer for BBOB-style f21 (Gallagher 101 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES optimizer for multimodal functions.
    
    Strategy:
    1. Use CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for global search.
    2. Adapt step size and covariance matrix to the landscape.
    3. Robust against local minima due to stochastic population-based search.
    4. Explicit deadline guard for 330s wall-clock limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_left():
            return deadline - time.time()

        best_x = None
        best_f = float('inf')
        call_count = 0

        def eval_point(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget or time_left() < 0.1:
                return None
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                return None
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        nu = 2 + 0.5 * mu
        c_sigma = (mu_eff + 2) / ((dim + 2) * 2 + mu_eff)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1) - 1)) + c_sigma
        c_cov = (5 + mu_eff / dim - mu_eff / 2) / ((dim + 2) * 3 + mu_eff)
        
        # Initialize
        x0 = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 + 1.0/(4*dim) - 1/(20*dim**2))
        
        pop = []
        fit = []
        
        # Initial population
        for i in range(lambda_):
            z = rng.standard_normal(dim)
            x = x0 + sigma * (C @ z)
            fx = eval_point(x)
            if fx is None:
                return best_x, best_f
            pop.append(x)
            fit.append(fx)
            
        indices = np.argsort(fit)
        pop = [pop[i] for i in indices]
        fit = [fit[i] for i in indices]
        
        # Main loop
        while call_count < budget and time_left() > 1.0:
            m_new = sum(weights[i] * pop[i] for i in range(mu))
            
            if sigma > 1e-12:
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C @ (m_new - x0) / sigma)
                h_sigma = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2/(dim+1)) * chi_n else 0.0
                pc = (1 - c_c) * pc + h_sigma * c_c * (m_new - x0)
                
                # Rank-1 + Rank-mu update
                C = (1 - c_sigma * c_sigma * d_sigma - c_cov * (1 - c_sigma * c_sigma)) * C \
                    + c_sigma * c_sigma * d_sigma * np.outer(pc, pc) \
                    + c_cov * sum(weights[i] * np.outer(pop[i] - x0, pop[i] - x0) / (sigma**2) for i in range(mu))
                
                C = (C + C.T) / 2
                
                # Check positive definiteness
                try:
                    eigvals = np.linalg.eigvalsh(C)
                    if np.min(eigvals) < 1e-6:
                        C = np.eye(dim)
                        sigma = 1.0
                        pc = np.zeros(dim)
                        ps = np.zeros(dim)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    sigma = 1.0
                    pc = np.zeros(dim)
                    ps = np.zeros(dim)
                
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps)/chi_n - 1))
            else:
                break
            
            x0 = m_new
            
            new_pop = []
            new_fit = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                x = x0 + sigma * (C @ z)
                fx = eval_point(x)
                if fx is None:
                    break
                new_pop.append(x)
                new_fit.append(fx)
                
            if len(new_pop) < lambda_:
                break
                
            indices = np.argsort(new_fit)
            pop = [new_pop[i] for i in indices]
            fit = [new_fit[i] for i in indices]
            
            # Convergence check
            if sigma < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
