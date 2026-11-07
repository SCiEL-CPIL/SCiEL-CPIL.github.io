import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None
        self.deadline = None

    def __call__(self, func):
        self.start_time = time.time()
        # Safety margin: 290 seconds out of 300 allowed
        self.deadline = self.start_time + 290.0
        
        d = self.dim
        budget = self.budget
        
        # Initial guess: center of the box
        x0 = np.zeros(d)
        # Initial sigma: scale to box size
        # Box is [-5, 5], so width 10. Minimizer in [-4, 4].
        sigma = 1.0
        
        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(d))
        if lam < 4: lam = 4
        # Weights
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # State
        x = x0.copy()
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        best_x = x0.copy()
        best_f = func(best_x)
        evals = 1
        
        # If we run out of budget immediately, return
        if budget <= 1:
            return best_x, best_f

        # Main loop
        while evals < budget:
            # Time check
            if time.time() > self.deadline:
                break
            
            # Generate offspring
            # Cholesky of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback if C becomes singular
                L = np.eye(d)
            
            # Sample
            z = np.random.randn(lam, d)
            # Apply transformation: x + sigma * z @ L.T
            # Note: C = L @ L.T, so sqrt(C) is L. We want z * sqrt(C).
            # If we generate z ~ N(0, I), then z @ L.T is ~ N(0, L.T @ L = C).
            # Wait, standard CMA-ES uses x + sigma * (L @ z).
            # Let's stick to z @ L.T or L @ z. Both are valid if symmetric.
            # Let's use: offspring = x + sigma * (z * np.sqrt(np.diag(C))) is wrong.
            # Correct: y = x + sigma * (L @ z)
            
            offspring = []
            for i in range(lam):
                step = L @ z[i]
                y = x + sigma * step
                y = np.clip(y, -5, 5)
                offspring.append(y)
            
            # Evaluate
            offspring_f = []
            for y in offspring:
                if evals >= budget:
                    break
                fy = func(y)
                offspring_f.append(fy)
                evals += 1
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            if not offspring_f:
                break
                
            # Sort
            indices = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in indices]
            sorted_offspring_f = [offspring_f[i] for i in indices]
            
            # Update mean
            x_new = np.sum([weights[i] * sorted_offspring[i] for i in range(mu)], axis=0)
            
            # Update sigma
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * L @ (x_new - x)
            # Note: The term is (x_new - x) / sigma
            diff = (x_new - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L @ diff)
            
            # Update pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * evals/lam))) < (1 + 2 / (d + 1)) * chi_d
            pc = (1 - c_c) * pc + hsig * c_c * (x_new - x)
            
            # Update C
            # delta_hsig = 1 - hsig
            delta_hsig = 1.0 if not hsig else 0.0
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C \
                + c_1 * np.outer(pc, pc) \
                + c_mu * np.sum([weights[i] * np.outer(sorted_offspring[i] - x, sorted_offspring[i] - x) for i in range(mu)], axis=0)
            
            # Regularization for C
            # Ensure C is positive definite
            # Simple check
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small identity
                C += 1e-6 * np.eye(d)
                
            x = x_new
            # Update sigma
            sigma *= np.exp(c_sigma / chi_d * (np.linalg.norm(ps) / chi_d - 1))
            
            # Clip sigma to prevent explosion
            if sigma < 1e-6:
                sigma = 1e-6
            if sigma > 10:
                sigma = 10

        return best_x, best_f
