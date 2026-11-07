import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Leave 10 seconds margin for safety
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline

        # CMA-ES Parameters
        # For dim=5, lambda=10 is a good balance
        lam = 10
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (20 * dim ** 2))
        
        # State variables
        # Initial guess: Origin
        x = np.zeros(dim)
        sigma = 2.0  # Start with a larger sigma to explore the [-5,5]^5 space
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        best_x = x.copy()
        best_f = func(x)
        calls = 1
        
        # Restart logic state
        last_improvement_call = 0
        stagnation_limit = 200  # If no improvement in 200 calls, consider restart
        
        while calls < budget and time_ok():
            # Generate population
            y_list = []
            f_list = []
            
            # Compute B and D from C for sampling
            # For dim=5, eigh is cheap
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.clip(eigenvalues, 1e-12, None)
            D = np.sqrt(eigenvalues)
            B = eigenvectors
            
            for i in range(lam):
                if not time_ok() or calls >= budget:
                    break
                z = np.random.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                # Clip to box if necessary, though func clips
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                y_list.append(y)
                f_list.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    last_improvement_call = calls
            
            if len(y_list) < mu:
                break
                
            # Sort by fitness
            idx = np.argsort(f_list)
            y_selected = np.array([y_list[i] for i in idx[:mu]])
            f_selected = np.array([f_list[i] for i in idx[:mu]])
            
            # Weighted mean
            m = np.sum(weights[:, None] * y_selected, axis=0)
            
            # Update evolution paths
            # ps is the sigma-normalized step
            old_sigma = sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * (m - x)))
            
            # Update pc
            # h_sigma: check if ps is small
            # Note: (calls - dim) might be negative or zero initially, handle carefully
            gen_count = max(1, calls // lam)
            h_sigma = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * gen_count)) < (1.4 + 2 / (dim + 1)) * chi_n
            
            if h_sigma:
                pc = (1 - c_c) * pc + c_c * (m - x) / sigma
                delta_h = 1 - h_sigma
            else:
                pc = (1 - c_c) * pc
                delta_h = 1
            
            # Update Covariance Matrix
            C = (1 - c_c * (2 - c_c + delta_h)) * C + c_c * (2 - c_c + delta_h) * (pc @ pc.T)
            C = (C + C.T) / 2.0
            
            # Update Sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update Mean
            x = m
            
            # Safety check: if sigma is too large, reduce it
            if sigma > 5.0:
                sigma = 5.0
            if sigma < 1e-4:
                break

            # Restart Logic
            # If we have made significant progress but stagnated, and we still have budget,
            # we can restart from the best point with a perturbed mean and larger sigma.
            if calls - last_improvement_call > stagnation_limit and calls < budget * 0.7:
                # Restart
                x = best_x + np.random.normal(0, 0.5, dim)
                x = np.clip(x, -5, 5)
                sigma = 1.0
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                last_improvement_call = calls
                # Don't reset best_x, best_f

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
        
        return best_x, best_f
