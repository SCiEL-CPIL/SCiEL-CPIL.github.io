import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = 2 * (4 + int(np.log(dim)))
        mu = lam // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1 / np.sum(w**2)
        
        # Learning rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)))
        
        # Damping
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Precompute square root of C for sampling: C = B D^2 B^T
        # Initially B=I, D=1
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        
        # We have 1 evaluation used.
        # Loop for remaining budget
        for _ in range(budget - 1):
            # Generate population
            ys = np.zeros((lam, dim))
            for i in range(lam):
                # Sample z ~ N(0, I)
                z = np.random.randn(dim)
                # y = x_mean + sigma * B * D * z
                ys[i] = x_mean + sigma * (B @ (D * z))
            
            # Clip to bounds
            ys = np.clip(ys, -5, 5)
            
            # Evaluate
            fs = np.array([func(y) for y in ys])
            
            # Update best
            idx_min = np.argmin(fs)
            if fs[idx_min] < best_f:
                best_f = fs[idx_min]
                best_x = ys[idx_min].copy()
            
            # Sort
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]
            
            # Update mean
            x_new = np.sum(w[:, None] * ys_sorted[:mu], axis=0)
            
            # Update covariance matrix adaptation
            # h_sigma condition
            norm_ps = np.linalg.norm(ps)
            h_sigma = (np.sqrt((1 + 2 / (c_sigma + 1)) - 1) * c_sigma + 1) * chi_n / np.sqrt(dim)
            
            if norm_ps / chi_n < (1.4 + 2 / (dim + 1)) * h_sigma:
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / chi_n * (B @ D @ (x_new - x_mean) / sigma)
            else:
                ps = (1 - c_sigma) * ps
            
            pc = (1 - c_c) * pc + h_sigma * (1 - c_c) * (x_new - x_mean) / sigma
            
            # Rank-one update
            y = (x_new - x_mean) / sigma
            C = (1 + c_1 * (c_mu * np.outer(y, y) - 1) - c_1) * C + c_mu * np.outer(y, y)
            
            # Rank-mu update
            y_mu = (ys_sorted[:mu] - x_mean) / sigma
            C = C + c_mu * np.sum(w[:, None, None] * (y_mu[:, :, None] * y_mu[:, None, :]), axis=0)
            
            # Re-estimate B and D
            # C is symmetric positive definite (usually)
            # Check for numerical stability
            if np.any(np.diag(C) <= 0) or not np.all(np.isfinite(C)):
                # Fallback: reset C to identity if it breaks
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)
            else:
                eigvals, eigvecs = np.linalg.eigh(C)
                # Clip small negative eigenvalues
                eigvals = np.clip(eigvals, 1e-12, None)
                D = np.sqrt(eigvals)
                B = eigvecs
                
            # Update sigma
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update mean
            x_mean = x_new
            
            # Optional: Keep mean within bounds? 
            # Standard CMA-ES doesn't, but if mean is far out, sampling is inefficient.
            # However, clipping the samples handles the evaluation.
            # If x_mean is out of bounds, the gradient estimation is biased.
            # For this task, starting at 0 and budget 500, it should stay reasonable.
            
        return best_x, best_f
