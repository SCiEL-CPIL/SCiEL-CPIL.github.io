import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        B = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        def eval_func(x):
            nonlocal calls
            if calls >= B:
                return np.inf
            calls += 1
            val = func(x)
            return val

        # CMA-ES Parameters
        lam = 4 + int(3 * np.log2(d))
        if lam > B: lam = B
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1 / np.sum(w**2)
        
        x_mean = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        dsigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        def get_cholesky(C):
            try:
                return np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                return np.eye(d)

        L = get_cholesky(C)
        
        while calls < B:
            pop = np.zeros((lam, d))
            for i in range(lam):
                if calls + i >= B:
                    break
                z = rng.standard_normal(d)
                pop[i] = x_mean + sigma * (L @ z)
                pop[i] = np.clip(pop[i], -5, 5)
            
            # Determine actual number of points in this generation
            n_gen = 0
            for i in range(lam):
                if calls + i >= B:
                    n_gen = i
                    break
            else:
                n_gen = lam
            
            if n_gen == 0:
                break
                
            # Evaluate
            fvals = np.zeros(n_gen)
            for i in range(n_gen):
                fvals[i] = eval_func(pop[i])
                
            if calls >= B:
                break
            
            # Sort
            idx = np.argsort(fvals)
            pop = pop[idx]
            fvals = fvals[idx]
            
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pop[0].copy()
            
            # Update mean
            x_new_mean = np.zeros(d)
            for i in range(min(mu, n_gen)):
                x_new_mean += w[i] * pop[i]
            
            # Update sigma
            step = (x_new_mean - x_mean) / sigma
            step_len = np.linalg.norm(step)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * step
            sigma *= np.exp(c_sigma / dsigma * (step_len / np.sqrt(d) - 1))
            
            # Update p_c
            h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2)) < (1 + c_sigma) * np.sqrt(d + 1) + 1e-8 else 0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * step
            
            # Update C
            term1 = p_c @ p_c.T
            if h_sigma == 0:
                term1 = term1 + (1 - h_sigma) * c_c * (2 - c_c) * C
            
            term2 = np.zeros((d, d))
            for i in range(min(mu, n_gen)):
                y_i = (pop[i] - x_mean) / sigma
                term2 += w[i] * np.outer(y_i, y_i)
            
            C = (1 - c_1 - c_mu) * C + c_1 * term1 + c_mu * term2
            C = (C + C.T) / 2
            
            x_mean = x_new_mean
            L = get_cholesky(C)

        return best_x, best_f
