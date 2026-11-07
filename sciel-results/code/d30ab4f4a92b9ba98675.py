import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        # Wrapper to count calls and enforce deadline
        def wrapper(x):
            nonlocal calls_made, best_f, best_x
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            
            calls_made += 1
            x = np.clip(x, -5, 5)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # CMA-ES Parameters
        lambda_ = 10  # Population size
        mu = lambda_ // 2  # Number of selected points
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1 / np.sum(w**2)
        
        # Step size
        sigma = 2.0
        # Mean
        mean = np.zeros(dim)
        # Covariance matrix
        C = np.eye(dim)
        # Eigen decomposition cache
        D = 1.0
        B = np.eye(dim)
        
        # Path adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff)))
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        def update_eigen():
            nonlocal D, B
            eigvals, eigvecs = np.linalg.eigh(C)
            # Sort in descending order
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            # Ensure positive eigenvalues
            eigvals = np.maximum(eigvals, 1e-12)
            D = 1.0 / np.sqrt(eigvals)
            B = eigvecs

        update_eigen()

        try:
            while calls_made < total_budget and time.time() < deadline:
                # Sample population
                Y = np.random.multivariate_normal(np.zeros(dim), np.eye(dim), size=lambda_)
                # Transform to X
                X = np.zeros((lambda_, dim))
                for i in range(lambda_):
                    X[i] = mean + sigma * (B @ (D * Y[i]))
                
                # Evaluate
                F = np.zeros(lambda_)
                for i in range(lambda_):
                    if calls_made >= total_budget or time.time() > deadline:
                        break
                    F[i] = wrapper(X[i])
                
                if calls_made >= total_budget or time.time() > deadline:
                    break
                
                # Sort by fitness
                idx = np.argsort(F)
                X = X[idx]
                F = F[idx]
                
                # Update mean
                new_mean = np.sum(w[:, np.newaxis] * X[:mu], axis=0)
                
                # Update covariance
                y = np.zeros((mu, dim))
                for i in range(mu):
                    y[i] = (X[i] - mean) / sigma
                
                # Update path covariance
                p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * (B @ (D @ (w @ y)))
                
                # Update path sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D @ (w @ y)))
                
                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                
                # Update covariance
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(p_c, p_c) - C)
                C += c_mu * np.sum(w[:, np.newaxis, np.newaxis] * np.outer(y, y), axis=0)
                
                # Symmetrize
                C = (C + C.T) / 2
                
                # Update mean
                mean = new_mean
                
                # Update eigen decomposition
                update_eigen()
                
                # Check convergence
                if sigma < 1e-8:
                    break
                    
        except TimeoutError:
            pass
        except Exception:
            pass

        return best_x, best_f
