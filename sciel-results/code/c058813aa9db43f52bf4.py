"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for the BBOB bent cigar problem."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        d = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # For d=5, lambda=4 is a good choice. 
        # mu = lambda / 2 = 2 (top 2 individuals)
        lam = 4
        mu = lam // 2
        
        # Weights for the top mu individuals
        # Standard weights: w_i = ln(mu + 0.5) - ln(i) for i=1..mu
        weights = np.array([np.log(mu + 0.5) - np.log(i) for i in range(1, mu + 1)])
        weights /= np.sum(weights)
        
        # Effective population size
        mueff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
        # c_sigma: control path length for sigma adaptation
        c_sigma = (mueff + 2) / (d + mueff + 5)
        # c_c: control path length for covariance adaptation
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        # c_1: rank-1 update weight
        c_1 = 2 / ((d + 1.3)**2 + mueff)
        # c_mu: rank-mu update weight
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        
        # Damping for covariance update
        # chi_d: expected norm of N(0, I)
        chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        # Initial state
        # Start at center or random? Random is safer for unknown shifts, but 
        # bent cigar is often centered. Let's start at 0. 
        # Actually, the problem says "hidden seeded shift". 
        # Starting at 0 is a good heuristic for BBOB instances usually.
        # However, random start in [-5, 5] is more robust if shift is large.
        # Given budget 500 and dim 5, CMA-ES is fast. 
        # Let's start at 0. If it fails, it will move.
        # But wait, "minimizer lies in [-4, 4]^5". 0 is inside.
        x = np.zeros(d)
        
        # Initial sigma
        # Heuristic: range / 4 ? Or 1.0?
        # For [-5,5], 1.0 is reasonable.
        sigma = 1.0
        
        # Covariance matrix
        C = np.eye(d)
        # Cholesky factor
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(d)
        
        # Paths
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Evaluation budget management
        # We use a population of size lam.
        # Total evaluations = lam * generations.
        # Reserve last 50 for polish.
        polish_budget = min(50, budget // 10)
        main_budget = budget - polish_budget
        
        # Number of full generations we can afford
        # We will loop until we run out of budget or time.
        
        best_x = x.copy()
        best_fx = func(best_x)
        calls = 1
        
        rng = np.random.default_rng(42) # Fixed seed for reproducibility in evaluation context? 
        # The evaluator creates a new instance for each call, so internal seed is fine.
        
        while calls + lam <= main_budget:
            if time.time() > deadline:
                break
                
            # 1. Sample population
            Y = np.zeros((lam, d))
            for i in range(lam):
                z = rng.standard_normal(d)
                Y[i] = x + sigma * np.dot(L, z)
            
            # Evaluate population
            F = np.zeros(lam)
            for i in range(lam):
                # Clip to box? The problem says func clips. 
                # "func(x) clips a finite length-5 vector to the box and returns a float."
                # So we don't need to clip explicitly, but it's good practice 
                # to ensure we don't pass NaNs or inf if something goes wrong.
                F[i] = func(Y[i])
                calls += 1
                
                if F[i] < best_fx:
                    best_fx = F[i]
                    best_x = Y[i].copy()
            
            # 2. Sort by fitness
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            F_sorted = F[idx]
            
            # 3. Update mean
            # Weighted sum of top mu
            x_new = np.dot(weights, Y_sorted[:mu])
            
            # 4. Update sigma (path adaptation)
            # Compute h_sigma: check if p_sigma is significant
            # h_sigma = (1 - exp(-2/(tau_sigma))) where tau_sigma = 1/sqrt(c_sigma)
            # Approximation:
            tau_sigma = 1.0 / np.sqrt(c_sigma)
            h_sigma = 1.0 - np.exp(-2.0 / tau_sigma)
            
            # Update p_sigma
            # p_sigma = (1-c_sigma)*p_sigma + sqrt(c_sigma*(2-c_sigma)*h_sigma) * (1/sigma) * L^-1 * (x_new - x)
            diff = x_new - x
            try:
                L_inv_diff = np.linalg.solve(L, diff)
            except np.linalg.LinAlgError:
                L_inv_diff = diff # Fallback
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * h_sigma) * (L_inv_diff / sigma)
            
            # Check condition for h_sigma
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls // lam + 1))) < (1.4 + 2/(d+1)) * chi_d:
                h_sigma = 1.0
            else:
                h_sigma = 0.0
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_d * (np.linalg.norm(p_sigma) / chi_d - 1))
            # Clip sigma to avoid explosion or zero
            sigma = np.clip(sigma, 1e-6, 100.0)
            
            # 5. Update covariance matrix
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c)) * (x_new - x) / sigma
            
            # Rank-1 update
            delta_sig = h_sigma * (np.linalg.norm(p_sigma) / chi_d - 1)
            update_rank1 = np.outer(p_sigma, p_sigma) + delta_sig * C
            
            # Rank-mu update
            # Y_sorted[:mu] are the top mu steps
            # We need the steps in the standard normal space? No, CMA-ES uses the actual steps.
            # The formula is: sum w_i * (y_i/sigma) (y_i/sigma)^T
            # But y_i = x_new - x? No, y_i are the sampled points.
            # The steps are Y_sorted[:mu] - x.
            Y_diff = (Y_sorted[:mu] - x) / sigma
            update_rank_mu = np.dot(weights, Y_diff[:, :, None] * Y_diff[:, None, :])
            
            # Combine
            C = (1 + c_1 * (h_sigma * delta_sig - 1) + c_mu * (h_sigma - 1)) * C \
                - c_1 * update_rank1 \
                - c_mu * update_rank_mu
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Check positive definiteness and update Cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If not PD, regularize
                C = C + 1e-10 * np.eye(d)
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Reset if completely broken
                    C = np.eye(d)
                    L = np.eye(d)
                    sigma = 1.0
                    p_sigma = np.zeros(d)
                    p_c = np.zeros(d)
            
            # Update mean
            x = x_new
            
            # Safety check on sigma
            if sigma < 1e-12:
                break

        # Polish phase
        if calls < budget and time.time() < deadline:
            from scipy.optimize import minimize
            try:
                # Use a tight tolerance
                result = minimize(func, best_x, method='Nelder-Mead', 
                                 options={'maxiter': budget - calls, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = result.x
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
