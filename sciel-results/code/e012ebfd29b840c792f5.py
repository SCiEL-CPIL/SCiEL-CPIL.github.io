"""Combined CMA-ES with sigma-restart strategy for Lunacek bi-Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 285.0  # Safety margin for 300s limit
        
        d = self.dim
        budget = self.budget
        
        # Global best tracking
        best_x = np.zeros(d)
        best_f = np.inf
        evals_used = 0
        
        def time_left():
            return timeout - (time.time() - start_time)

        # CMA-ES Parameters
        # Population size: lambda = 4 + 3*ln(d) is standard, but for d=5, lambda=10 is good.
        # We use a fixed lambda for stability in short budgets.
        lam = 10 
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        
        # Chi_d: Expected norm of standard normal in d dimensions
        chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(32*d*d))
        
        # Initial state
        m = np.zeros(d)
        sigma = 2.0  # Cover significant portion of [-5, 5]
        pc = np.zeros(d)
        C = np.eye(d)
        B = np.eye(d)
        D = np.ones(d)
        
        # Evaluate initial point
        f_init = func(m)
        evals_used += 1
        if f_init < best_f:
            best_f = f_init
            best_x = m.copy()
            
        last_improvement_eval = evals_used
        stagnation_threshold = 4 * lam  # Wait 4 generations without improvement
        
        while evals_used < budget and time_left() > 1.0:
            # If remaining budget is too small for a full generation, do (1+1) steps
            if budget - evals_used < lam:
                # (1+1)-ES polish
                while evals_used < budget and time_left() > 0.5:
                    y = m + sigma * np.random.randn(d)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    evals_used += 1
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        sigma *= 1.05
                    else:
                        sigma *= 0.95
                        if sigma < 1e-4:
                            break
                break
            
            # Generate population
            X = np.array([m + sigma * B @ (D * np.random.randn(d)) for _ in range(lam)])
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            evals_used += lam
            
            # Update best
            idx_best = np.argmin(F)
            if F[idx_best] < best_f:
                best_f = F[idx_best]
                best_x = X[idx_best].copy()
                last_improvement_eval = evals_used
            
            # Sort
            idx = np.argsort(F)
            x_sorted = X[idx]
            
            # Weighted mean
            m_new = sum(w[i] * x_sorted[i] for i in range(mu))
            
            # Update pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (B.T @ ((m_new - m) / sigma))
            
            # Sigma update
            # Standard CMA-ES sigma update
            sigma *= np.exp(c_sigma * (np.linalg.norm(pc) / chi_d - 1))
            
            # Update C
            delta_h = 1.0 if np.linalg.norm(pc) / chi_d < (1.4 + 2/(d+1)) else 0.0
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) + (1 - (1 - c_c)**2) * C)
            for i in range(mu):
                diff = x_sorted[i] - m_new
                C += c_mu * w[i] * np.outer(diff, diff)
            
            # Symmetrize and decompose
            C = (C + C.T) / 2
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = (eigvecs * eigvals) @ eigvecs.T
                B = eigvecs
                D = np.sqrt(eigvals)
            except np.linalg.LinAlgError:
                # Fallback
                B = np.eye(d)
                D = np.ones(d)
                C = np.eye(d)
                
            m = m_new
            
            # Stagnation Check and Restart Strategy
            # If no improvement for stagnation_threshold evals, restart
            # Strategy: Reset C to Identity, keep m at best_x, increase sigma to explore new basins
            if evals_used - last_improvement_eval >= stagnation_threshold:
                # Restart
                m = best_x.copy()
                sigma = max(sigma * 1.5, 1.0)  # Expand search space
                pc = np.zeros(d)
                C = np.eye(d)
                B = np.eye(d)
                D = np.ones(d)
                last_improvement_eval = evals_used
                
                # Optional: If we have restarted many times with large sigma and still stuck, 
                # we might want to try a smaller sigma to refine. But for Lunacek, 
                # finding the other basin is key.
                
        return best_x, best_f
# EVOLVE-BLOCK-END
