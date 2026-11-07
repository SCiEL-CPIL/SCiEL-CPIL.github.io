import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_x = None
        best_f = float('inf')
        
        def timed_call(x):
            nonlocal n_calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            val = func(x)
            n_calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return float(val)

        # 1. Random Initialization to find a good starting point
        n_init = 20
        init_points = []
        for _ in range(n_init):
            if n_calls >= budget:
                break
            x = np.random.uniform(-5, 5, dim)
            f = timed_call(x)
            init_points.append((f, x.copy()))
        
        if n_calls >= budget:
            return best_x, best_f

        init_points.sort(key=lambda x: x[0])
        x_start = init_points[0][1].copy()
        
        # 2. CMA-ES Optimization
        # Parameters
        lambda_ = 4 + int(3 * np.log(dim))  # Population size
        mu = lambda_ // 2                     # Number of recombination weights
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        nu = 2 * weights[0] * (1 - weights[0])  # Variance reduction factor
        
        # Adaptation parameters
        c_sigma = (mu + 2) / (dim + mu + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * ((mu + 2) / (dim + 2) ** 2 - 1))
        
        # Initial state
        x_m = x_start.copy()
        sigma = 2.0  # Initial step size, large enough to explore [-5,5]
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute eigendecomposition of C (initially I, so B=I, D=1)
        # We will update this every generation
        
        while n_calls < budget and time.time() < deadline:
            # Generate population
            ys = []
            fs = []
            
            for i in range(lambda_):
                if n_calls >= budget or time.time() > deadline:
                    break
                # Sample y from N(0, C)
                y = B @ (D * np.random.standard_normal(dim))
                x = x_m + sigma * y
                f = timed_call(x)
                ys.append(y)
                fs.append(f)
                
            if len(ys) < lambda_:
                break
                
            # Sort by fitness
            indices = np.argsort(fs)
            ys = [ys[i] for i in indices]
            fs = [fs[i] for i in indices]
            
            # Update mean
            y_w = sum(weights[i] * ys[i] for i in range(mu))
            x_m = x_m + sigma * y_w
            
            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * nu) * (B.T @ (D * (B @ y_w))) / sigma if sigma > 1e-10 else p_sigma
            # Note: The term (B.T @ (D * (B @ y_w))) is essentially y_w transformed back to the original space? 
            # Actually, standard CMA-ES update for p_sigma is:
            # p_sigma <- (1-c_sigma)p_sigma + sqrt(c_sigma(2-c_sigma)nu) * (1/sigma) * B.T * (D * (B * y_w))
            # Wait, y_w is already in the transformed space? No, y_w is the weighted sum of y's (which are N(0,C)).
            # Let's stick to the standard formulation.
            # y_w is the weighted mean of the sampled y's.
            # The update for p_sigma is:
            # p_sigma = (1-c_sigma)*p_sigma + sqrt(c_sigma*(2-c_sigma)*nu) * (B.T * (D * (B * y_w))) / sigma
            # This is correct.
            
            # Correction: The term B.T * (D * (B * y_w)) is the inverse transformation of y_w?
            # If y = B D z, then z = D^{-1} B.T y.
            # The standard update is p_sigma <- (1-c_sigma)p_sigma + sqrt(...) * (1/sigma) * (B.T * (D * (B * y_w)))
            # Let's verify. If C=I, B=I, D=1. Then term is y_w/sigma.
            # p_sigma tracks the evolution path in the original coordinate system?
            # Actually, p_sigma is in the "original" space of the step, but scaled.
            
            # Update sigma
            if sigma > 1e-10:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) - chi_n))
            # chi_n is expected norm of N(0, I)
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            # Update covariance matrix
            # p_c update
            y_w_sigma = y_w / sigma if sigma > 1e-10 else np.zeros(dim)
            p_c = (1 - c_c) * p_c + c_c * (1.0 if np.linalg.norm(p_sigma) > np.sqrt((1 - (1 - c_sigma)**2) * (2 - c_sigma) * nu) else 0.0) * y_w_sigma
            
            # Heuristic for h_sigma (delta of Heaviside function)
            h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2/(dim+1)) * np.sqrt(1 - (1 - c_sigma)**2) * np.sqrt(1 - (1 - c_sigma)**2) * 1.0 else 0.0
            # Simplified h_sigma check:
            # h_sigma = 1 if ||p_sigma|| / sqrt(1-(1-c_sigma)^2) < (1.4 + 2/(dim+1)) * chi_n * sqrt(1-(1-c_sigma)^2)
            # Actually, standard condition:
            # h_sigma = 1 if ||p_sigma||/sqrt(1-(1-c_sigma)^2) < (1.4 + 2/(dim+1)) * sqrt(1-(1-c_sigma)^2) * chi_n
            # Let's use a simpler robust check:
            h_sigma = 1.0 if np.linalg.norm(p_sigma) < (1.4 + 2/(dim+1)) * np.sqrt(1 - (1 - c_sigma)**2) * chi_n else 0.0
            
            # Update p_c with h_sigma
            p_c = (1 - c_c) * p_c + c_c * h_sigma * y_w_sigma
            
            # Update C
            rank_one = np.outer(p_c, p_c)
            rank_mu = sum(weights[i] * np.outer(ys[i], ys[i]) for i in range(mu))
            
            C = (1 + c_1 * h_sigma * (c_c * (2 - c_c) - c_1) * np.outer(p_c, p_c) - c_1) * C + c_1 * rank_mu
            
            # Wait, the formula is:
            # C <- (1 + c_1 * h_sigma * (c_c * (2 - c_c) - c_1) * p_c * p_c^T - c_1) * C + c_1 * sum w_i * y_i * y_i^T
            # My previous line was slightly off. Let's correct.
            C = (1 - c_1) * C + c_1 * (h_sigma * (1 + c_c * (2 - c_c)) * np.outer(p_c, p_c) - np.eye(dim)) + c_mu * rank_mu
            # Actually, standard CMA-ES update:
            # C <- (1 + c_1 * h_sigma * (c_c * (2 - c_c) - c_1) * p_c * p_c^T - c_1) * C + c_1 * sum_{i=1}^mu w_i * y^{(i)} * y^{(i)T}
            # Let's use this standard form.
            C = (1 + c_1 * h_sigma * (c_c * (2 - c_c) - c_1) * np.outer(p_c, p_c) - c_1) * C + c_1 * rank_mu
            
            # Eigendecomposition
            D2, B = np.linalg.eigh(C)
            D2 = np.sqrt(np.maximum(D2, 1e-12))
            D = D2
            
            # Check for convergence
            if sigma < 1e-10:
                break

        return best_x, best_f
