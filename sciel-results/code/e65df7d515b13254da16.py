"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        
        # Parameters
        n = self.dim
        lambda_ = max(4, int(4 + 3 * np.log(n)))  # Population size
        mu = lambda_ // 2  # Number of selected individuals
        
        # Initial state
        x_mean = np.zeros(n)
        sigma = 2.0  # Initial step size
        C = np.eye(n)  # Covariance matrix
        pc = np.zeros(n)  # Evolution path for covariance
        ps = np.zeros(n)  # Evolution path for step size
        
        # CMA-ES parameters
        cmu = 2 / ((n + 1.3) ** 2 + 10)
        c1 = (2 + 2.0 / (n + 1)) / ((n + 2) ** 2 + 5)
        cmuc = cmu + 2 * c1 * (1 - c1)
        c_sigma = (n + 1.8) / (n + 9)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (n + 1)) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1 / np.sum(w**2)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_s = (2 + 2 * min(1.0, (mu_eff - 1) / n)) / (n + 2 + 2 * min(1.0, (mu_eff - 1) / n))
        
        best_x = x_mean
        best_f = np.inf
        calls_made = 0
        
        # Initial population evaluation
        # We need to be careful with the budget. 
        # Budget is total calls. 
        # Initial: lambda_ calls.
        # If lambda_ > budget, we must adjust.
        if lambda_ >= self.budget:
             # Fallback to simple random search or (1+1)-ES if budget is tiny
             # But budget is 500, so this won't happen.
             pass

        # Precompute sqrt(C) using Cholesky
        try:
            B = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            B = np.eye(n)
        
        # Main Loop
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Generate population
            # We need to check if we have enough budget for a full generation
            # If not, we can stop or do a partial generation (less effective for CMA-ES)
            # Since CMA-ES relies on full generations for update, we stop if remaining < lambda_
            if self.budget - calls_made < lambda_:
                break
                
            Y = np.zeros((lambda_, n))
            F = np.zeros(lambda_)
            
            for i in range(lambda_):
                z = np.random.randn(n)
                y = x_mean + sigma * (B @ z)
                # Clip to box [-5, 5]
                y = np.clip(y, -5, 5)
                Y[i] = y
                F[i] = func(y)
                calls_made += 1
                
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = y.copy()
            
            # Sort by fitness
            idx = np.argsort(F)
            Y_sel = Y[idx[:mu]]
            F_sel = F[idx[:mu]]
            
            # Update mean
            w_sum = np.sum(w)
            x_new = np.sum(w[:, np.newaxis] * Y_sel, axis=0)
            
            # Update step size and paths
            # Re-center Y_sel for path update? 
            # Standard CMA-ES uses the step from the old mean.
            # y_i = x_mean + sigma * B * z_i
            # So B * z_i = (y_i - x_mean) / sigma
            # We need the z vectors or the steps.
            # Let's recompute steps from Y_sel
            steps = (Y_sel - x_mean) / sigma
            
            # Evolution path for step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ (w @ steps)) / chi_n
            
            # Evolution path for covariance
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (w @ steps)
            
            # Update covariance
            delta_hs = (np.linalg.norm(ps)**2 / chi_n**2 > (1 + 4 / n) * (1 - c_sigma))
            rank_mu = np.outer(w @ steps, w @ steps)
            # Note: The standard formula uses the steps in the "z-space" or "y-space"?
            # C = (1 - c1 - cmu) * C + c1 * (outer(pc, pc) + (delta_hs - 1 + c_c) * I) + cmu * sum(wi * outer(yi, yi))
            # Where yi are the selected steps (y_i - x_mean)/sigma ? No, usually the steps are the deviations.
            # Let's use the standard update with the steps (deviations normalized by sigma? No, deviations are y_i - x_mean)
            # Actually, the update is on the covariance of the steps z.
            # But we generated y = x_mean + sigma * B * z.
            # So z = B^-1 * (y - x_mean) / sigma.
            # It is easier to work with the y-steps.
            # Let y_steps = (Y_sel - x_mean) / sigma. These are the "z" vectors in the transformed space?
            # No, y = x_mean + sigma * B * z. So (y - x_mean) / sigma = B * z.
            # The covariance C is updated based on z.
            # z_i = B^-1 * (y_i - x_mean) / sigma.
            
            # Let's compute z_sel
            z_sel = np.linalg.solve(B, steps.T).T  # B * z = steps => z = B^-1 * steps
            
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                rank_mu += w[i] * np.outer(z_sel[i], z_sel[i])
                
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (delta_hs - 1 + c_c) * np.eye(n)) + cmu * rank_mu
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update B
            try:
                B = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If not positive definite, reset or project
                # Simple fix: add small identity
                C += 1e-6 * np.eye(n)
                B = np.linalg.cholesky(C)
                
            x_mean = x_new
            
            # Stop criterion: if sigma is too small or convergence is reached
            if sigma < 1e-12:
                break
            if np.all(np.abs(Y_sel - Y_sel[0]) < 1e-8):
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
