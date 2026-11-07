import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0
        d = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Lambda: population size
        lam = 4 + int(3 * np.log(d))
        # Cutoff mu
        mu = lam // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        # Learning rates
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = 2 / ((d + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff)))
        
        # Initialize
        x_mean = np.zeros(d)
        sigma = 1.0
        C = np.eye(d)
        B = np.eye(d)
        D = np.ones(d)
        
        # PC and PS for path evolution
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # We need to track the best x and f found globally.
        best_x = None
        best_f = np.inf
        
        n_evals = 0
        it = 0
        
        # Phase 1: CMA-ES with larger population for exploration
        while n_evals + lam <= budget and time.time() - start_time < timeout:
            # Sample
            Z = np.random.randn(lam, d)
            Y = x_mean + sigma * (Z * D) @ B.T
            Y = np.clip(Y, -5, 5)
            
            # Evaluate
            F = np.array([func(y) for y in Y])
            n_evals += lam
            
            # Update best
            min_idx = np.argmin(F)
            if F[min_idx] < best_f:
                best_f = F[min_idx]
                best_x = Y[min_idx].copy()
                
            # Sort
            sorted_idx = np.argsort(F)
            Y_sorted = Y[sorted_idx]
            
            # Update mean
            y_old = x_mean.copy()
            x_mean = np.dot(w, Y_sorted[:mu])
            
            # Path evolution sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (B.T @ (D * (x_mean - y_old)))
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (it + 1)))) < (1 + 2 / (d + 1)) * cc
            it += 1
            
            # Path evolution covariance
            if hsig:
                pc = (1 - cc) * pc + cc / sigma * (B.T @ (D * (x_mean - y_old)))
            
            # Update covariance
            delta_hebbian = c1 * (np.outer(ps, ps) + (1 - hsig) * (cc * (2 - cc) * np.outer(pc, pc)))
            
            for i in range(mu):
                delta_hebbian += (cmu * w[i]) * np.outer(Y_sorted[i] - y_old, Y_sorted[i] - y_old) / (sigma * sigma)
                
            C = (1 - c1 - cmu) * C + delta_hebbian
            C = (C + C.T) / 2
            
            D2, B = np.linalg.eigh(C)
            D = np.sqrt(np.clip(D2, 1e-20, None))
            
            # Step size adaptation
            chiN = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
            sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Stop CMA-ES if sigma is very small or we've used 70% of budget
            if sigma < 1e-5 or n_evals > 0.7 * budget:
                break

        # Phase 2: Local refinement using Nelder-Mead if we have budget left
        if best_x is not None and n_evals < budget and time.time() - start_time < timeout:
            # Use remaining budget for local search
            remaining = budget - n_evals
            # Nelder-Mead with limited iterations
            from scipy.optimize import minimize
            try:
                res = minimize(
                    func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': remaining, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except:
                pass

        # Final check
        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)
            
        return best_x, best_f
