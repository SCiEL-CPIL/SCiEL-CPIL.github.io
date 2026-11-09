import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 330s limit
        
        best_x = None
        best_f = np.inf
        call_count = 0
        rng = np.random.default_rng(42)
        
        def wrapped_func(x):
            nonlocal best_x, best_f, call_count
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if call_count >= self.budget:
                raise StopIteration("Budget exceeded")
            
            fx = func(x)
            call_count += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            return fx

        # Phase 1: Latin Hypercube Sampling for exploration
        # Use ~80 LHS points in [-4, 4]^dim to quickly locate the basin
        n_lhs = min(80, self.budget // 6)
        lhs_points = []
        for i in range(n_lhs):
            x = np.zeros(self.dim)
            for d in range(self.dim):
                stratum = (i + rng.random()) / n_lhs
                x[d] = -4.0 + 8.0 * stratum
            lhs_points.append(x)
        
        rng.shuffle(lhs_points)
        
        for x in lhs_points:
            if time.time() > deadline or call_count >= self.budget:
                break
            try:
                wrapped_func(x)
            except (TimeoutError, StopIteration):
                break
        
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            call_count = 1
        
        # Phase 2: CMA-ES refinement from the best point
        remaining_budget = self.budget - call_count
        if remaining_budget > 10 and best_x is not None:
            n = self.dim
            pop_size = max(4, int(4 + 3 * np.log(n)))  # ~9 for dim=5
            lambda_ = pop_size
            
            # Initialize CMA-ES state
            mean = best_x.copy()
            sigma = 1.0
            C = np.eye(n)  # Covariance matrix
            C_inv = np.eye(n)
            C_sqrt = np.eye(n)  # Cholesky factor
            
            # Evolution paths
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            
            # Strategy parameters
            mu = lambda_ // 2  # Number of recombination weights
            # Weights: rank-based
            w = np.zeros(mu)
            for i in range(mu):
                w[i] = np.log(mu + 0.5) - np.log(i + 1)
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w**2)
            
            # Learning rates
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((n + 2)**2 + mu_eff))
            
            # Damping for sigma update
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n**2))
            
            # Stagnation counter
            stagnation = 0
            best_cma_f = best_f
            
            while call_count < self.budget and time.time() < deadline:
                if call_count + lambda_ > self.budget:
                    # Do a partial generation if not enough budget for full pop
                    lambda_ = self.budget - call_count
                    if lambda_ < 2:
                        break
                    mu = max(1, lambda_ // 2)
                    w = np.zeros(mu)
                    for i in range(mu):
                        w[i] = np.log(mu + 0.5) - np.log(i + 1)
                    w /= w.sum()
                    mu_eff = 1.0 / np.sum(w**2)
                    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
                    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
                    c_1 = 2 / ((n + 1.3)**2 + mu_eff)
                    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2.0 / mu_eff) / ((n + 2)**2 + mu_eff))
                
                # Sample population
                y = np.zeros((lambda_, n))
                z = np.zeros((lambda_, n))
                for i in range(lambda_):
                    z[i] = rng.standard_normal(n)
                    y[i] = mean + sigma * (C_sqrt @ z[i])
                    y[i] = np.clip(y[i], -5.0, 5.0)
                
                # Evaluate population
                fy = np.zeros(lambda_)
                for i in range(lambda_):
                    if call_count >= self.budget or time.time() > deadline:
                        lambda_ = i
                        break
                    try:
                        fy[i] = wrapped_func(y[i])
                    except (TimeoutError, StopIteration):
                        lambda_ = i
                        break
                
                if lambda_ < 2:
                    break
                
                # Sort by fitness
                order = np.argsort(fy[:lambda_])
                y = y[order]
                fy = fy[order]
                z = z[order]
                
                # Update best
                if fy[0] < best_f:
                    best_f = fy[0]
                    best_x = y[0].copy()
                
                # Recombination
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += w[i] * y[i]
                
                # Update mean
                mean = mean + c_c * (y_w - mean)
                
                # Update evolution path for covariance
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_w / sigma)
                
                # Update evolution path for sigma
                p_sigma = (1 - c_c_sigma) * p_sigma if False else (1 - c_sigma) * p_sigma
                p_sigma += np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (y_w / sigma)
                
                # Update covariance matrix
                C = (1 + c_1 * (c_mu * mu_eff - 1) - c_1) * C
                C += c_1 * (np.outer(p_c, p_c) - C)
                for i in range(mu):
                    C += c_mu * w[i] * np.outer(y[i] / sigma, y[i] / sigma)
                
                # Symmetrize and ensure positive definiteness
                C = (C + C.T) / 2
                try:
                    C_inv, C_sqrt = self._update_chol(C, C_inv, C_sqrt)
                except:
                    # If Cholesky fails, reset covariance
                    C = np.eye(n)
                    C_inv = np.eye(n)
                    C_sqrt = np.eye(n)
                
                # Update sigma
                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**20) < (1.4 + 2 / (n + 1)) * chi_n)
                if hsig:
                    p_c = (1 - c_c) * p_c
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_w / sigma)
                
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                
                # Check for stagnation
                if fy[0] >= best_cma_f:
                    stagnation += 1
                else:
                    stagnation = 0
                    best_cma_f = fy[0]
                
                # If stagnating for too long, try a larger jump
                if stagnation > 3 * lambda_:
                    mean = np.clip(best_x + rng.standard_normal(n) * 0.5, -5.0, 5.0)
                    sigma = max(sigma, 0.5)
                    C = np.eye(n)
                    C_inv = np.eye(n)
                    C_sqrt = np.eye(n)
                    p_sigma = np.zeros(n)
                    p_c = np.zeros(n)
                    stagnation = 0
        
        # Phase 3: Final polish with very small steps
        remaining_budget = self.budget - call_count
        if remaining_budget > 5 and best_x is not None:
            local_best_x = best_x.copy()
            local_best_f = best_f
            local_sigma = 0.05
            
            for step in range(min(remaining_budget, 100)):
                if call_count >= self.budget or time.time() > deadline:
                    break
                
                y = np.clip(local_best_x + local_sigma * rng.standard_normal(self.dim), -5.0, 5.0)
                fy = wrapped_func(y)
                
                if fy < local_best_f:
                    local_best_x = y.copy()
                    local_best_f = fy
                    local_sigma *= 0.95
                else:
                    local_sigma *= 0.9
                
                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()
        
        return best_x, best_f
    
    def _update_chol(self, C, C_inv, C_sqrt):
        """Update Cholesky factorization of covariance matrix."""
        n = C.shape[0]
        # Use Cholesky decomposition
        try:
            L = np.linalg.cholesky(C)
            C_inv = np.linalg.inv(L @ L.T)
            C_sqrt = L
            return C_inv, C_sqrt
        except np.linalg.LinAlgError:
            # Add small regularization
            C_reg = C + 1e-10 * np.eye(n)
            L = np.linalg.cholesky(C_reg)
            C_inv = np.linalg.inv(L @ L.T)
            C_sqrt = L
            return C_inv, C_sqrt
