import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls = 0
        
        # CMA-ES Parameters
        # Standard settings for dim=5
        lambda_ = 4 * dim + 2
        mu = lambda_ // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))
        
        # Initial state
        m = np.zeros(dim) # Start at center
        sigma = 1.0       # Initial step size, box is [-5, 5]
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        best_x = m.copy()
        best_f = float('inf')
        
        def wrapper(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            return float(func(x_clipped))

        # Main loop
        while calls + lambda_ <= budget:
            # 1. Sample
            B = np.random.multivariate_normal(np.zeros(dim), C, size=lambda_)
            # Actually, standard CMA-ES samples y ~ N(0, C), x = m + sigma * y
            # We can do: y = np.random.randn(lambda_, dim) @ np.linalg.cholesky(C).T
            L = np.linalg.cholesky(C)
            Y = np.random.randn(lambda_, dim)
            X = m + sigma * (Y @ L.T)
            
            # Evaluate
            F = np.array([wrapper(x) for x in X])
            
            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()
                
            # Sort
            I = np.argsort(F)
            Xs = X[I]
            Fs = F[I]
            
            # 2. Update Mean
            m_new = np.sum(weights[:, None] * Xs[:mu], axis=0)
            
            # 3. Update Covariance
            y = (m_new - m) / sigma
            pc = (1 - c_c) * pc + c_c * y
            hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c)**(2 * (calls + 1)))) < (1 + 2 / (dim + 1)) * chi_n
            ps = (1 - c_sigma) * ps + (np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) if hsig else 0) * y
            
            # Rank-mu update
            v = Xs[:mu] - m
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 if hsig else 0)) + c_mu * np.sum(weights[:, None, None] * v[:, :, None] * v[:, None, :], axis=0)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Regularize if needed
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-4 * np.eye(dim)
                L = np.linalg.cholesky(C)
                
            # 4. Update Step Size
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            m = m_new
            # Clip mean to box? No, CMA-ES handles it via sigma reduction usually, 
            # but clipping the mean helps keep search in bounds.
            m = np.clip(m, -5, 5)

        # If budget remains, do a few random local steps or just return
        return best_x, best_f
