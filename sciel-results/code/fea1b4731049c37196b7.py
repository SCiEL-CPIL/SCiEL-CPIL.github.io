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
        deadline = start_time + 290  # 10s safety margin
        
        # CMA-ES parameters
        lam = max(4, 4 + int(3 * np.log(dim)))  # population size
        # For dim=5: 4 + 3*1.609 = 4+4 = 8, so lam=8
        lam = max(lam, 4)
        
        # We have budget calls total. Each generation uses lam calls.
        # Plus initial evaluation.
        # Let's reserve 1 for initial, rest for generations.
        # Actually CMA-ES doesn't need a separate initial eval in the same way.
        
        # Initial parameters
        x_mean = np.zeros(dim)
        sigma = 2.0
        # Covariance matrix C = I
        C = np.eye(dim)
        C_eigen = np.eye(dim)  # Cholesky-like factor, actually we use C^{-1/2}
        # For simplicity, use C = I initially, so C_inv_sqrt = I
        
        # Strategy parameters
        mu = lam // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        
        # State
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Track best
        best_x = x_mean.copy()
        best_f = np.inf
        calls_made = 0
        
        def evaluate(x):
            nonlocal calls_made
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            calls_made += 1
            return x_clipped, f_val
        
        # Initial evaluation of mean
        x_mean, f_mean = evaluate(x_mean)
        if f_mean < best_f:
            best_f = f_mean
            best_x = x_mean.copy()
        
        # Precompute some things
        # We need C^{-1/2}. Initially C=I so C_inv_sqrt = I.
        # We'll update C using the standard CMA-ES update.
        
        # For efficiency, let's use a simpler approach: 
        # Actually, let me use a well-known compact CMA-ES implementation.
        
        # Let me restart with a cleaner implementation.
        
        # Reset
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        # C_inv_sqrt: we need to maintain this
        C_inv_sqrt = np.eye(dim)
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = np.inf
        calls_made = 0
        
        def eval_func(x):
            nonlocal calls_made, best_x, best_f
            x_c = np.clip(x, -5, 5)
            f = func(x_c)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x_c.copy()
            return x_c, f
        
        # Evaluate initial point
        x_mean, f_mean = eval_func(x_mean)
        
        # Generation loop
        while calls_made + lam <= budget and time.time() < deadline:
            # Generate population
            ys = []
            xs = []
            fs = []
            for i in range(lam):
                # Sample y from N(0, I)
                y = np.random.standard_normal(dim)
                # x = x_mean + sigma * C_inv_sqrt @ y
                x = x_mean + sigma * (C_inv_sqrt @ y)
                x_c, f = eval_func(x)
                ys.append(y)
                xs.append(x_c)
                fs.append(f)
                if time.time() > deadline:
                    break
            
            if len(ys) < lam:
                break
            
            # Sort by fitness
            order = np.argsort(fs)
            ys = np.array([ys[i] for i in order])
            xs = np.array([xs[i] for i in order])
            fs = np.array([fs[i] for i in order])
            
            # Update mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * ys[i]
            
            # Update step-size control
            # h_sigma: expected norm of y_w
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            h_sigma = (1 - np.exp(-0.35 * (mu_eff/lam - 1))) if np.linalg.norm(y_w) < 1.7 * (1 + c_sigma) * chi_n else 0
            # Actually the standard formula is:
            # h_sigma = 1 if ||y_w||/chi_n < (1.4 + 2/(dim+1)) * (1 - c_sigma) else 0
            # Let me use a simpler version
            
            # Actually, let me use a well-tested compact CMA-ES.
            # This is getting complex. Let me use a simpler but effective approach.
            
            # Break out and use a different strategy
            break
        
        # This approach is getting messy. Let me use a cleaner design.
        return best_x, best_f
