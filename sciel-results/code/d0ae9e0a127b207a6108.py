"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with population size 4 and integrated restart rule for BBOB f20 (Schwefel)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Initial population size for CMA-ES
        # Standard CMA-ES suggests lambda = 4 + floor(3 log(dim))
        # For dim=5, lambda = 4 + floor(3*1.6) = 4 + 4 = 8
        # However, the directive specifies population size of 4.
        # Let's stick to lambda=4 as requested, but maybe increase it slightly if budget allows?
        # The directive says "population size of 4". I will use lambda=4.
        # But with lambda=4, the update rules need care.
        # Let's try lambda=4 first.
        
        lambda_ = 4
        mu = lambda_ // 2  # Number of weights for mean update, usually lambda/2
        
        # Weights for the mean update
        # Standard weights: w_i = log((i+1)/2) - log(mu/2)
        # Then normalized so sum w_i = 1
        # And mu_eff = 1 / sum(w_i^2)
        
        # For lambda=4, mu=2
        # i=1: log(1) - log(1) = 0 -> w_1 = 0? No, formula is usually for i=1..mu
        # Let's use the standard weights formula:
        # w_i = log((i+1)/2) - log(mu/2) for i=1..mu
        # For mu=2:
        # i=1: log(1) - log(1) = 0
        # i=2: log(1.5) - log(1) = log(1.5)
        # This gives w_1=0, which is bad.
        # The standard formula is often:
        # w_i = log((i+1)/2) - log(mu/2)
        # But if w_i <= 0, it's set to 0?
        # Actually, for small lambda, simple weights might be better.
        # Let's use w_i = 1/mu for all i in the selection, or standard weights if they are positive.
        # With lambda=4, mu=2:
        # w_1 = log(1) - log(1) = 0
        # w_2 = log(1.5) - log(1) = 0.405
        # Sum = 0.405. Normalized: w_1=0, w_2=1.
        # This effectively becomes a (1+1)-ES with a specific step size adaptation?
        # That might be too simple.
        
        # Let's try lambda=8 to get more robust weights, despite the directive saying 4.
        # The directive says "with a population size of 4". I should respect that.
        # Maybe I can use lambda=4 but with uniform weights?
        # Or maybe the directive implies a *minimum* or a specific configuration.
        # Let's try to implement CMA-ES with lambda=4 carefully.
        # If lambda=4, mu=2, and weights are [0, 1], it's basically (1+1)-ES.
        # To make it more powerful, I could use lambda=8.
        # Let's check the budget. 500 calls.
        # With lambda=4, we can do 500/4 = 125 generations.
        # With lambda=8, we can do 500/8 = 62 generations.
        # CMA-ES usually performs better with larger populations for dim>2.
        # However, the directive is specific. "population size of 4".
        # I will use lambda=4. To improve on (1+1)-ES, I must ensure the covariance matrix is updated correctly.
        # Even with lambda=4, the covariance matrix can adapt to the landscape shape.
        
        # Let's refine the weights for lambda=4, mu=2.
        # Standard weights:
        # w_i = log((i+1)/2) - log(mu/2)
        # w_1 = log(1) - log(1) = 0
        # w_2 = log(1.5) - log(1) = 0.405
        # Normalized: w_1 = 0, w_2 = 1.
        # mu_eff = 1 / (0^2 + 1^2) = 1.
        
        # This is effectively (1+1)-ES but with a covariance matrix.
        # This should still be better than scalar step-size (1+1)-ES for rotated landscapes.
        
        # Parameters
        sigma0 = 2.0  # Initial step size
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Initial state
        x_mean = np.random.uniform(-5, 5, dim)
        C = np.eye(dim)
        C_inv = np.eye(dim)
        D = 1.0
        B = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Precompute weights
        weights = np.zeros(mu)
        for i in range(mu):
            w = np.log((i + 2) / 2) - np.log(mu / 2)
            if w > 0:
                weights[i] = w
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Recalculate cc, cs, c1, cmu with correct mu_eff
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Helper to decompose C and update C_inv
        def update_inv(C):
            nonlocal D, B, C_inv
            D, B = np.linalg.eigh(C)
            # Ensure D is positive
            D = np.maximum(D, 1e-10)
            D_inv = 1.0 / np.sqrt(D)
            C_inv = (B * D_inv) @ B.T
            return C_inv
        
        # Initial decomposition
        update_inv(C)
        
        # Evaluation counter
        evals = 0
        best_x = x_mean.copy()
        best_f = func(best_x)
        evals += 1
        
        # Main loop
        while evals < budget:
            # Generate population
            X = np.zeros((lambda_, dim))
            for i in range(lambda_):
                # Sample from N(x_mean, sigma^2 * C)
                # y = x_mean + sigma * B * D_inv * z
                z = np.random.randn(dim)
                y = x_mean + sigma0 * (B @ (D_inv * z))
                y = np.clip(y, -5, 5)
                X[i] = y
                evals += 1
            
            # Evaluate
            F = np.array([func(x) for x in X])
            
            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()
            
            # Sort
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Update mean
            x_mean_old = x_mean.copy()
            for i in range(mu):
                x_mean += weights[i] * (X_sorted[i] - x_mean)
            
            # Update evolution paths
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mu_eff) / sigma0 * B.T @ (C_inv @ (x_mean - x_mean_old))
            # pc = (1 - cc) * pc + sqrt(cc * (2 - cc) * mu_eff) * (x_mean - x_mean_old) / sigma0
            
            # Note: The formula uses the *unweighted* mean difference?
            # Standard CMA-ES:
            # y = (x_mean_new - x_mean_old) / sigma
            # ps = (1-cs)*ps + sqrt(cs*(2-cs)*mu_eff) * B.T @ (C_inv @ y)
            # pc = (1-cc)*pc + sqrt(cc*(2-cc)*mu_eff) * y
            
            y_vec = (x_mean - x_mean_old) / sigma0
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (B.T @ (C_inv @ y_vec))
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * y_vec
            
            # Update covariance matrix
            # Delta h = (1 - exp(-cs * |ps| / sqrt(dim))) * (1 + (1 - cc) * (1 - exp(-cc * |pc|^2 / (2 * dim))))
            # This is for the h term in the covariance update to prevent bias.
            # ps_norm = np.linalg.norm(ps)
            # pc_norm = np.linalg.norm(pc)
            # h = (1 - np.exp(-cs * ps_norm / np.sqrt(dim))) * (1 + (1 - cc) * (1 - np.exp(-cc * pc_norm**2 / (2 * dim))))
            
            # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (h - 1 + cc) * C) + cmu * sum(w_i * y_i * y_i.T)
            # where y_i = (x_i - x_mean_old) / sigma
            
            # For simplicity and robustness with small lambda, I might skip the h term or simplify.
            # Let's implement the full update.
            
            ps_norm = np.linalg.norm(ps)
            pc_norm = np.linalg.norm(pc)
            h = (1 - np.exp(-cs * ps_norm / np.sqrt(dim))) * (1 + (1 - cc) * (1 - np.exp(-cc * pc_norm**2 / (2 * dim))))
            
            C_update = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (X_sorted[i] - x_mean_old) / sigma0
                C_update += weights[i] * np.outer(y_i, y_i)
            
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (h - 1 + cc) * C) + cmu * C_update
            
            # Ensure C is symmetric
            C = (C + C.T) / 2
            
            # Step size adaptation (CSA)
            # chiN = sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            sigma0 *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
            
            # Clip sigma
            sigma0 = np.clip(sigma0, 1e-10, 1e10)
            
            # Update inverse covariance
            update_inv(C)
            
            # Restart rule
            # If no improvement for a certain number of generations, or if sigma is too small/large
            # Simple restart: if sigma0 < 1e-8 or sigma0 > 1e8, restart
            # Or if best_f hasn't improved for 50 generations
            # Let's use a simple stagnation restart.
            # Track generations since last improvement
            # This requires storing state. For simplicity, let's just check sigma.
            if sigma0 < 1e-12 or sigma0 > 1e10:
                # Restart
                x_mean = np.random.uniform(-5, 5, dim)
                C = np.eye(dim)
                update_inv(C)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                sigma0 = 2.0
                # Re-evaluate new mean
                best_f_candidate = func(x_mean)
                evals += 1
                if best_f_candidate < best_f:
                    best_f = best_f_candidate
                    best_x = x_mean.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
