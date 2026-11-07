"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """CMA-ES with hybrid restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        total_calls = 0
        dim = self.dim
        budget = self.budget
        
        # Weierstrass is multimodal. CMA-ES is good for unimodal/rotated.
        # Strategy: Run several short CMA-ES restarts from different locations.
        # If time allows, do a final local polish.
        
        # Initial points: Random + Center
        # Since minimizer is in [-4, 4]^5, center is a good guess, but random is safer for multimodal.
        # Let's try 5-10 restarts depending on time.
        
        num_restarts = 0
        max_restarts = 10
        
        # Estimate time per call? No, just check deadline in loop.
        
        while total_calls < budget and time.time() < deadline and num_restarts < max_restarts:
            
            # Choose starting point
            if num_restarts == 0:
                # First restart: center
                x0 = np.zeros(dim)
            else:
                # Subsequent restarts: random within [-4, 4]
                x0 = rng.uniform(-4, 4, dim)
            
            # Check if we have enough budget for a meaningful run
            remaining_budget = budget - total_calls
            if remaining_budget < 10:
                break
                
            # CMA-ES parameters
            # Population size: 4 + floor(3 log dim) -> 4 + 3*1.6 = 8 for dim=5
            lambda_pop = int(4 + 3 * np.log(dim))
            if lambda_pop > remaining_budget:
                lambda_pop = max(2, remaining_budget // 2)
                
            # Initial sigma: 1.0 is reasonable for [-5,5]
            sigma = 1.0
            
            x = x0.copy()
            fx = func(x)
            total_calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            # CMA-ES state
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            # Evolution path lengths
            cmacov = 1.0 / (4 + 0.38 * dim + np.log(1 + 2/dim))
            mu = lambda_pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1 / np.sum(weights**2)
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 2 / mueff) / ((dim + 2) ** 2 + mueff))
            chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            # Main loop for this restart
            while total_calls < budget and time.time() < deadline:
                # Generate offspring
                B = np.linalg.cholesky(C)
                z = rng.standard_normal((lambda_pop, dim))
                y = np.dot(z, B.T)
                ys = y / np.linalg.norm(y, axis=1, keepdims=True)
                
                # Update paths
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * weights[:mu, None] @ y[:mu] / sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (total_calls % 10000 + 1)))) < (1.4 + 2 / (dim + 1)) * chiN
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * np.mean(y[:mu], axis=0)
                
                # Update covariance
                deltaD = (hsig - 1) * c_c * (2 - c_c)
                C = (1 + c_1 * deltaD - c_1 - c_mu) * C + \
                    c_1 * (np.outer(pc, pc) - C) + \
                    c_mu / sigma**2 * (weights[:mu, None] * y[:mu]) @ (weights[:mu, None] * y[:mu]).T
                
                # Ensure positive definite
                C = (C + C.T) / 2
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    B = np.linalg.cholesky(C + np.eye(dim) * 1e-4)
                    
                # Generate candidates
                candidates = []
                for i in range(lambda_pop):
                    xi = x + sigma * np.dot(y[i], B.T)
                    xi = np.clip(xi, -5, 5)
                    candidates.append(xi)
                    
                # Evaluate
                fvals = []
                for xi in candidates:
                    if time.time() > deadline:
                        break
                    if total_calls >= budget:
                        break
                    fxi = func(xi)
                    total_calls += 1
                    fvals.append(fxi)
                    if fxi < best_f:
                        best_f = fxi
                        best_x = xi.copy()
                        
                if len(fvals) < lambda_pop:
                    break
                    
                # Sort
                idx = np.argsort(fvals)
                y_sorted = y[idx]
                
                # Update mean
                x = x + sigma * (weights[:mu, None] @ y_sorted[:mu])
                
                # Update sigma
                sigma *= np.exp(c_sigma / chiN * (np.linalg.norm(y_sorted[:mu, 0]) / np.sqrt(mueff) - 1))
                
                # Check for stagnation or convergence
                if sigma < 1e-6:
                    break
                if np.all(np.abs(x - np.clip(x, -5, 5)) < 1e-8):
                    # Stuck at boundary, maybe restart
                    break
                    
            num_restarts += 1
            
        # Final local polish if time permits
        if best_x is not None and time.time() < deadline - 10 and total_calls < budget:
            try:
                res = minimize(lambda v: func(v), best_x, method='Nelder-Mead', 
                               options={'maxiter': 50, 'fatol': 1e-8, 'xatol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass
                
        return best_x, best_f
# EVOLVE-BLOCK-END
