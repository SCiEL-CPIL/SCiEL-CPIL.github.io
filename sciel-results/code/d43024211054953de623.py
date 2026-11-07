import numpy as np
import time

class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for BBOB f13 (Sharp ridge).
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        total_evals = 0
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        cs = (2 + mueff / (2 * dim)) / (dim + 2 + mueff / 2)
        dsigma = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) + cs
        
        max_restarts = 5
        restart_count = 0
        
        while total_evals < budget and restart_count < max_restarts:
            if time.time() > deadline:
                break
            
            # Initialize state for this restart
            if best_x is not None:
                m = best_x.copy()
            else:
                m = rng.uniform(-2, 2, dim)
            
            C = np.eye(dim)
            sigma = 1.0 if restart_count == 0 else 1.0 + 0.5 * restart_count
            
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            gen = 0
            stagnation_counter = 0
            last_best_f = np.inf
            
            while total_evals < budget:
                if time.time() > deadline:
                    break
                
                if total_evals + lam > budget:
                    break

                # Sample population
                Y = []
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = C + 1e-8 * np.eye(dim)
                    L = np.linalg.cholesky(C)
                
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    y = m + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    Y.append(y)
                
                # Evaluate population
                F = []
                for y in Y:
                    if total_evals >= budget:
                        break
                    fy = func(y)
                    F.append(fy)
                    total_evals += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                
                if len(F) < lam:
                    break
                
                idx = np.argsort(F)
                Y_sorted = [Y[i] for i in idx]
                F_sorted = [F[i] for i in idx]
                
                m_new = m + sigma * np.sum(weights[:, np.newaxis] * np.array(Y_sorted[:mu]), axis=0)
                
                d = (m_new - m) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * d
                
                # hsig calculation
                norm_ps = np.linalg.norm(ps)
                if gen == 0:
                    hsig = 1.0
                else:
                    # Expected norm of ps
                    # E[||ps||] ~ sqrt(dim) * sqrt(1 - (1-cs)^gen) ... approx
                    # Standard check:
                    hsig = (norm_ps / np.sqrt(1 - (1 - cs)**(gen + 1))) < (1.4 + 2 / (dim + 1)) * cc * np.sqrt(1 - (1 - cs)**(gen + 1))
                    # Wait, the standard formula is:
                    # h_sigma = ||p_sigma|| / sqrt(1-(1-c_s)^{2g}) < (1.4 + 2/(D+1)) * c_c
                    # But p_sigma is scaled by sqrt(mu_eff) in the update?
                    # Let's use the simpler version from many implementations:
                    hsig = (norm_ps / np.sqrt(1 - (1 - cs)**(gen + 1))) < (1.4 + 2 / (dim + 1)) * cc
                
                pc = (1 - c1) * pc + hsig * np.sqrt(c1 * (2 - c1) * mueff) * d
                
                C = (1 - mu * c1 - cmu) * C + c1 * np.outer(ps, ps)
                for i in range(mu):
                    diff = Y_sorted[i] - m
                    C += cmu * weights[i] * np.outer(diff, diff)
                
                sigma *= np.exp((cs / dsigma) * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
                
                m = m_new
                gen += 1
                
                if F_sorted[0] < last_best_f:
                    last_best_f = F_sorted[0]
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                
                if stagnation_counter > 15 or sigma < 1e-8:
                    restart_count += 1
                    break
        
        return best_x, best_f
