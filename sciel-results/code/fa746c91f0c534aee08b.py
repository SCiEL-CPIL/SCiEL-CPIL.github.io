import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 280.0
        
        def time_ok():
            return time.time() < deadline

        # CMA-ES Parameters
        lam = 12  # Population size
        mu = lam // 2  # Number of recombination weights
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        w = np.concatenate([w, np.zeros(lam - mu)])
        
        mueff = 1.0 / np.sum(w**2)
        
        # Adaptation rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 - mueff / (dim + 2) + c_c) / ((dim + 1.3) ** 2 + mueff + c_c)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # Initialization
        x_mean = np.zeros(dim)
        # Start with a sigma that covers a good portion of the search space
        # Domain is [-5, 5], min in [-4, 4]. 
        # A sigma of 1.0 is a good start.
        sigma = 1.0 
        
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = np.inf
        evals_used = 0
        
        def evaluate(x):
            nonlocal best_f, best_x, evals_used
            x = np.clip(x, -5, 5)
            f = func(x)
            evals_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Initial evaluation of the mean
        f_mean = evaluate(x_mean)
        
        # Precompute Cholesky decomposition of C
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(dim)
        
        while evals_used < budget and time_ok():
            # Generate offspring
            ys = []
            xs = []
            fs = []
            
            for i in range(lam):
                if evals_used >= budget or not time_ok():
                    break
                
                z = np.random.randn(dim)
                y = L @ z
                x_new = x_mean + sigma * y
                x_new = np.clip(x_new, -5, 5)
                
                f_new = evaluate(x_new)
                
                ys.append(y)
                xs.append(x_new)
                fs.append(f_new)
                
            if not xs:
                break
            
            ys = np.array(ys)
            xs = np.array(xs)
            fs = np.array(fs)
            
            # Sort by fitness
            order = np.argsort(fs)
            xs = xs[order]
            ys = ys[order]
            fs = fs[order]
            
            # Update mean
            x_new_mean = np.dot(w[:mu], xs[:mu].T)
            
            # Update evolution paths
            # Check if we should update sigma based on ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / np.linalg.norm(L) * (L @ (np.dot(w[:mu], ys[:mu].T)))
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(2 * dim * (1 - np.exp(-2 / (dim + 1)))) - 1))
            
            # Update covariance matrix
            delta_sig = (1 + c_sigma) * (np.linalg.norm(ps) / np.sqrt(2 * dim * (1 - np.exp(-2 / (dim + 1)))) - 1)
            
            # Outer product term for covariance update
            # We need to compute the update to C
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (delta_sig**2 + c_sigma*(2-c_sigma)) * np.eye(dim)) + c_mu * sum(w_i * (y_i @ y_i.T))
            
            # First, update pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (x_new_mean - x_mean) / sigma
            
            # Compute the rank-1 update from pc
            pc_term = pc @ pc.T + (delta_sig**2 + c_sigma * (2 - c_sigma)) * np.eye(dim)
            
            # Compute the rank-mu update from the selected offspring
            # We need to use the ys (standard normal space) or xs?
            # Standard CMA-ES updates C in the standard normal space using ys, then transforms?
            # Actually, the standard update is:
            # C = (1 - c1 - cmu) C + c1 (pc pc^T + ...) + cmu sum w_i (y_i y_i^T)
            # where y_i are the steps in the standard normal space.
            
            mu_term = np.zeros((dim, dim))
            for i in range(mu):
                mu_term += w[i] * np.outer(ys[i], ys[i])
            
            C = (1 - c_1 - c_mu) * C + c_1 * pc_term + c_mu * mu_term
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Regularize if needed
            # Check eigenvalues
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C += 1e-10 * np.eye(dim)
            
            # Update Cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If it fails, reset C to identity and sigma to a small value?
                # Or just skip this generation's update?
                # For robustness, we can add a small diagonal
                C += 1e-6 * np.eye(dim)
                L = np.linalg.cholesky(C)
            
            # Update mean
            x_mean = x_new_mean
            
            # Convergence check
            # If sigma is very small or the spread of the population is tiny
            if sigma < 1e-10 or (np.max(fs[:mu]) - np.min(fs[:mu]) < 1e-10):
                break

        return best_x, best_f
