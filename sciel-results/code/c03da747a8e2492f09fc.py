import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        dim = self.dim
        total_budget = self.budget
        
        x_best = None
        f_best = np.inf
        
        rng = np.random.default_rng(42)
        
        used_calls = 0
        
        def check_deadline():
            return used_calls >= total_budget or time.time() > deadline
        
        # CMA-ES Parameters - tuned for dim=5
        N = 8
        mu = N // 2
        
        # Pre-compute weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Expected norm of standard normal vector
        from scipy.special import gamma
        c_sigma_norm = np.sqrt(2) * gamma((dim + 1) / 2) / gamma(dim / 2)
        
        # Strategy: 
        # 1. Initial broad exploration with CMA-ES centered at 0.
        # 2. If it stagnates, restart centered at best point with smaller sigma.
        # 3. Final polishing with scipy minimize.
        
        max_generations_per_run = 30
        min_improvement = 1e-8
        
        run_id = 0
        stagnation_count = 0
        last_best_f = np.inf
        no_improve_global = 0
        
        while not check_deadline():
            run_id += 1
            
            # Initialize CMA-ES state for this run
            if run_id == 1:
                m = np.zeros(dim)
                sigma = 2.5  # Broad initial search
            elif run_id == 2:
                m = x_best.copy()
                sigma = 1.0
            else:
                m = x_best.copy()
                sigma = max(0.1, 0.5 * (run_id - 2) ** 0.3)

            C = np.eye(dim)
            ps = np.zeros(dim)
            
            # Run CMA-ES for this phase
            gen = 0
            no_improve_in_phase = 0
            phase_best_f = f_best
            
            while not check_deadline():
                if gen >= max_generations_per_run:
                    break
                
                gen += 1
                
                # Sample population
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)
                
                # Generate N samples
                Z = rng.standard_normal((N, dim))
                X = m[np.newaxis, :] + sigma * (Z @ L.T)
                
                # Evaluate population
                F = np.zeros(N)
                for i in range(N):
                    if used_calls >= total_budget or time.time() > deadline:
                        break
                    x_trial = np.clip(X[i], -5, 5)
                    F[i] = func(x_trial)
                    used_calls += 1
                    
                    if F[i] < f_best:
                        f_best = F[i]
                        x_best = x_trial.copy()
                
                if used_calls >= total_budget or time.time() > deadline:
                    break
                
                # Sort by fitness
                idx = np.argsort(F)
                X_sel = X[idx[:mu]]
                
                # Update mean
                m_new = np.sum(weights[:, np.newaxis] * X_sel, axis=0)
                
                # Update step size (CSA)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (L.T @ (m_new - m))
                
                # Delta term for covariance update
                delta = 1.0 + 2 * max(0, (c_c - 1) / (1 - c_c ** (2 * (gen + 1)))) if c_c > 0 else 1.0
                # Actually, use the standard formula:
                delta = 1.0 + 2 * max(0, (c_c - 1) / (1 - c_c ** (2 * (gen + 1))))
                
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / c_sigma_norm - 1))
                sigma = max(sigma, 1e-10)
                
                # Update covariance
                y = (m_new - m) / sigma
                # Rank-1 update
                C = (1 + c_1 * delta - c_1 - c_mu) * C + c_1 * np.outer(y, y)
                
                # Rank-mu update
                Y = (X_sel - m) / sigma
                w2 = weights ** 2 / mu_eff
                # Sum of weighted outer products
                rank_mu_sum = np.zeros((dim, dim))
                for j in range(mu):
                    rank_mu_sum += w2[j] * np.outer(Y[j], Y[j])
                C += c_mu * rank_mu_sum
                
                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                C += 1e-8 * np.eye(dim)
                
                # Update mean
                m = m_new
                
                # Check for improvement in this phase
                current_phase_best = np.min(F)
                if current_phase_best < phase_best_f - min_improvement:
                    phase_best_f = current_phase_best
                    no_improve_in_phase = 0
                else:
                    no_improve_in_phase += 1
                    
                # Stagnation check within phase
                if no_improve_in_phase > 8:
                    break
                if sigma < 1e-8:
                    break
                
                # Check global stagnation
                if f_best < last_best_f - min_improvement:
                    last_best_f = f_best
                    stagnation_count = 0
                    no_improve_global = 0
                else:
                    stagnation_count += 1
                    no_improve_global += 1
                    
                if stagnation_count > 15:
                    break

        # Final polishing phase with scipy
        if x_best is not None and used_calls < total_budget and time.time() < deadline:
            try:
                def objective(x):
                    nonlocal used_calls
                    if used_calls >= total_budget or time.time() > deadline:
                        return f_best
                    x_clipped = np.clip(x, -5, 5)
                    val = func(x_clipped)
                    used_calls += 1
                    if val < f_best:
                        f_best = val
                        x_best = x_clipped.copy()
                    return val
                
                result = minimize(objective, x_best, method='Nelder-Mead', 
                                 options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            except Exception:
                pass

        return x_best, f_best
