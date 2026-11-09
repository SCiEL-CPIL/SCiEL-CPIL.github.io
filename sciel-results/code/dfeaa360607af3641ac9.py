import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 280

        def safe_func(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            self.calls += 1
            f_val = func(x)
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x.copy()
            return f_val

        n = self.dim
        # CMA-ES Parameters
        # Population size
        lambda_pop = 4 + int(3 * np.log(n))
        if lambda_pop < 10:
            lambda_pop = 10
        
        # Initial step size
        sigma = 2.0 
        # Initial mean: center of the box
        # Using 0 is a good default for BBOB if shift is small, but random is safer.
        # Let's use a deterministic random start based on seed 42 for reproducibility in tests?
        # The task says "hidden seeded shift", so the function is fixed per instance.
        # Using a fixed start is fine.
        rng = np.random.default_rng(42)
        mean = np.zeros(n)
        
        # Covariance matrix
        C = np.eye(n)
        
        # Evolution paths
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        # Time constants
        cs = (lambda_pop + 2) / (n + lambda_pop + 5)
        ds = 1 + 2 * max(0, np.sqrt((lambda_pop - 1) / (n + 1)) - 1) + cs
        cmu = min(1 / (n * lambda_pop), 0.2)
        cc = (4 + n / lambda_pop) / (n + 4 + 2 * lambda_pop / n)
        c_sigma = (lambda_pop / 2 + 1) / (n + lambda_pop + 5)
        
        # Weights
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        # Main Loop
        while self.calls < self.budget and time.time() < deadline:
            # Generate population
            # Sample y ~ N(0, C)
            # Using Cholesky decomposition for stability and speed for small n
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not PD, reset or use identity
                C = np.eye(n)
                L = np.eye(n)
                
            ys = rng.standard_normal((lambda_pop, n)) @ L.T
            xs = mean + sigma * ys
            
            # Evaluate population
            fs = []
            for x in xs:
                if self.calls >= self.budget or time.time() > deadline:
                    break
                f_val = safe_func(x)
                fs.append(f_val)
            
            if not fs:
                break
                
            fs = np.array(fs)
            
            # Sort by fitness
            order = np.argsort(fs)
            xs_sorted = xs[order]
            fs_sorted = fs[order]
            
            # Update mean
            new_mean = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
            
            # Update evolution paths
            # Compute C_inv and C_inv_sqrt
            try:
                C_inv = np.linalg.inv(C)
                C_sqrt = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(C_sqrt)
            except:
                C_inv = np.eye(n)
                C_inv_sqrt = np.eye(n)
                
            y_mean = (new_mean - mean) / sigma
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C_inv_sqrt @ y_mean)
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * (C_inv @ y_mean)
            
            # Update sigma
            hsig = np.linalg.norm(ps) < (1 + 1 / cs) * c_sigma * np.sqrt(n)
            sigma *= np.exp((c_sigma / ds) * (np.linalg.norm(ps) / np.sqrt(n) - 1))
            
            # Update C
            y_sigma = (1 + (hsig - 0.5) * cs * cc) * pc
            delta_hsig = (1 - hsig) * cs * (1 - cc)
            
            rank_mu_matrix = np.sum(weights[:, None, None] * np.outer(xs_sorted[:mu] - mean, xs_sorted[:mu] - mean), axis=0) / sigma**2
            rank_one_matrix = np.outer(y_sigma, y_sigma)
            
            C = (1 + cmu * (delta_hsig - 1) - cmu) * C + cmu * rank_mu_matrix + (cmu * delta_hsig) * rank_one_matrix
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update mean
            mean = new_mean
            
            # Clip mean to box to keep search relevant
            mean = np.clip(mean, -5, 5)
            
            # Check for stagnation or convergence
            if sigma < 1e-8:
                break
            if sigma > 50:
                sigma = 50

        return self.best_x, self.best_f
