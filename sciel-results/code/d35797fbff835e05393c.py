import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget
        deadline = start_time + 280  # safety margin before 300s limit

        # Use CMA-ES for robust optimization of the convex Sphere function
        # Start with a large sigma to cover the domain, CMA-ES will adapt
        best_x, best_f, calls = self._cmaes(func, np.zeros(n), 4.0, budget, deadline)
        
        # If we have budget left, we could try a second run with a smaller sigma
        # centered at the best point found, but for Sphere, CMA-ES usually converges well.
        # Given the budget is small (500) and dim is small (5), one good run is often best.
        # However, if the first run used too many calls without converging, a restart might help.
        # But typically CMA-ES is very efficient.
        
        return best_x, best_f

    def _cmaes(self, func, x0, sigma, budget, deadline):
        """
        Simple CMA-ES implementation.
        """
        n = len(x0)
        calls = 0
        best_x = x0.copy()
        best_f = float('inf')
        
        # Parameters
        # Default parameters for CMA-ES
        pop_size = int(4 + 3 * np.log(n))  # lambda = 4 + 3 ln(n)
        mu = pop_size // 2  # number of weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        
        # Initialize state
        x_mean = x0.copy()
        C = np.eye(n)
        C_sqrt = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        # Evaluate initial point
        f0 = func(x_mean)
        calls += 1
        if f0 < best_f:
            best_f = f0
            best_x = x_mean.copy()
            
        while calls < budget and time.time() < deadline:
            # Generate population
            ys = []
            xs = []
            fs = []
            
            for i in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Sample y
                z = np.random.randn(n)
                y = C_sqrt @ z
                x_new = x_mean + sigma * y
                
                # Evaluate
                f_new = func(x_new)
                calls += 1
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
                
                ys.append(y)
                xs.append(x_new)
                fs.append(f_new)
                
            if len(fs) < pop_size:
                break
                
            # Sort
            order = np.argsort(fs)
            ys = [ys[i] for i in order]
            xs = [xs[i] for i in order]
            fs = [fs[i] for i in order]
            
            # Update mean
            x_mean_new = x_mean + sigma * sum(weights[i] * ys[i] for i in range(mu))
            
            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (C_sqrt @ (x_mean_new - x_mean))
            h_sig = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls // pop_size))) < (1.4 + 2/(n+1)) * chi_n else 0.0
            pc = (1 - c_c) * pc + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_new - x_mean) / sigma
            
            # Update covariance
            delta_hsig = (1 - h_sig) * c_c * (2 - c_c)
            rank_one = np.outer(pc, pc)
            rank_mu = sum(weights[i] * np.outer(ys[i], ys[i]) for i in range(mu))
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure positive definite (regularization if needed)
            # Simple check: if eigenvalues become too small, reset or add epsilon
            # For dim 5, this is usually stable.
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update x_mean
            x_mean = x_mean_new
            
            # Update C_sqrt (Cholesky decomposition)
            try:
                # Symmetrize C to avoid numerical issues
                C = (C + C.T) / 2
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback: reset C to identity if it becomes singular
                C = np.eye(n)
                C_sqrt = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)
                sigma = max(sigma, 0.1) # Don't let sigma get too small immediately

            # Stop condition: if sigma is very small or no improvement
            if sigma < 1e-8:
                break
            if np.max(fs[:mu]) - np.min(fs[:mu]) < 1e-12 * (1 + abs(np.min(fs))):
                break

        return best_x, best_f, calls
