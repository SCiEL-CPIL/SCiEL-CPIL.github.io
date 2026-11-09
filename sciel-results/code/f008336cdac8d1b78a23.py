import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        n = self.dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        rng = np.random.default_rng(0)
        x_mean = rng.uniform(-5, 5, n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        evals = 1
        
        while evals < self.budget:
            if time.time() > deadline:
                break
            
            B = np.linalg.cholesky(C)
            z = rng.standard_normal((lam, n))
            y = z @ B.T * sigma
            x = x_mean + y
            
            f = np.array([func(xi) for xi in x])
            evals += lam
            
            order = np.argsort(f)
            x = x[order]
            f = f[order]
            
            if f[0] < best_f:
                best_f = f[0]
                best_x = x[0].copy()
                
            w = weights
            x_mean_new = np.dot(w, x[:mu])
            
            y_mean = (x_mean_new - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**((evals + 1) / c_sigma))) < (1.4 + 2 / (n + 1)) * np.sqrt(1 - (1 - c_c)**((evals + 1) / c_c))
            
            # Efficient update of C
            # C = (1 + c1*(hsig-1) - c1)*C + c1*outer(pc,pc) + c_mu * sum(wi * outer(xi-xmean, xi-xmean)) / sigma^2
            # Note: The sum term is the weighted outer product of the selected points relative to the old mean?
            # Standard CMA-ES uses the new mean for the rank-mu update?
            # Actually, the standard update is:
            # C = (1 - c1 - c_mu)*C + c1*outer(pc,pc) + c_mu * sum(wi * outer(yi, yi))
            # where yi are the selected steps (x_i - x_mean_old) / sigma?
            # Let's stick to the standard formulation:
            # The rank-mu update uses the vectors from the selected points to the *new* mean? Or old?
            # Standard: v_i = (x_i - x_mean_old) / sigma
            # C += c_mu * sum(w_i * outer(v_i, v_i))
            
            v = (x[:mu] - x_mean) / sigma
            rank_mu = np.sum([weights[i] * np.outer(v[i], v[i]) for i in range(mu)], axis=0)
            
            C = (1 + c_1 * (hsig - 1) - c_1) * C + c_1 * np.outer(pc, pc) + c_mu * rank_mu
            C = (C + C.T) / 2
            x_mean = x_mean_new
            
            if sigma < 1e-12:
                break

        return best_x, best_f
