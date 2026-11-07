import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        time_limit = 290  # 10s margin
        
        best_x = None
        best_f = np.inf
        evals_used = 0
        
        # Helper to evaluate and track
        def eval_func(x):
            nonlocal evals_used
            if evals_used >= self.budget:
                return np.inf
            val = func(x)
            evals_used += 1
            return val

        # CMA-ES implementation
        class CMAES:
            def __init__(self, x0, sigma, popsize=9):
                self.dim = len(x0)
                self.x = x0.copy()
                self.sigma = sigma
                self.lambda_ = popsize
                self.mu = self.lambda_ // 2
                
                w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
                self.weights = w / np.sum(w)
                self.mu_eff = 1.0 / np.sum(self.weights**2)
                
                cs = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
                ds = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1) + cs
                self.c_sigma = cs * ds
                self.ds = ds
                self.c_c = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
                self.c_1 = 2 / ((self.dim + 1.3)**2 + self.mu_eff)
                self.c_mu = min(1 - self.c_1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.dim + 2)**2 + self.mu_eff))
                
                self.C = np.eye(self.dim)
                self.C_inv = np.eye(self.dim) # Not strictly needed if we use Cholesky, but inv is easy for dim=5
                self.pc = np.zeros(self.dim)
                self.ps = np.zeros(self.dim)
                self.chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))
                
                self.best_x = x0.copy()
                self.best_f = np.inf
                self.gen = 0

            def step(self, eval_func):
                self.gen += 1
                y = []
                for i in range(self.lambda_):
                    # Use Cholesky for sampling? np.random.multivariate_normal uses SVD/Cholesky internally.
                    # For dim=5, it's fast.
                    z = np.random.multivariate_normal(np.zeros(self.dim), self.C)
                    x_new = self.x + self.sigma * z
                    y.append(x_new)
                
                f_vals = []
                for x_new in y:
                    f = eval_func(x_new)
                    if f < self.best_f:
                        self.best_f = f
                        self.best_x = x_new.copy()
                    f_vals.append(f)
                
                f_vals = np.array(f_vals)
                idx = np.argsort(f_vals)
                y = [y[i] for i in idx]
                
                # Update mean
                self.x = sum(self.weights[i] * y[i] for i in range(self.mu))
                
                # Update sigma and C
                y_norm = [ (y[i] - self.x) / self.sigma for i in range(self.mu) ]
                self.ps = (1 - self.c_sigma) * self.ps + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * (sum(self.weights[i] * y_norm[i] for i in range(self.mu)))
                
                hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.c_sigma)**(2 * self.gen))) < (1.4 + 2 / (self.dim + 1)) * self.chi_n
                
                self.pc = (1 - self.c_c) * self.pc + hsig * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * (sum(self.weights[i] * y_norm[i] for i in range(self.mu)))
                
                delta_hsig = (1 - hsig) * self.c_c * (2 - self.c_c)
                rank_one = np.outer(self.pc, self.pc)
                rank_mu = sum(self.weights[i] * np.outer(y_norm[i], y_norm[i]) for i in range(self.mu))
                
                self.C = (1 + self.c_1 * delta_hsig - self.c_1 - self.c_mu) * self.C + self.c_1 * rank_one + self.c_mu * rank_mu
                
                # Regularize C to ensure positive definite
                self.C = (self.C + self.C.T) / 2
                # If eigenvalues are too small, add identity
                # For dim=5, simple check
                eigvals = np.linalg.eigvalsh(self.C)
                if np.min(eigvals) < 1e-6:
                    self.C += 1e-6 * np.eye(self.dim)
                
                self.sigma *= np.exp(self.c_sigma / self.ds * (np.linalg.norm(self.ps) / self.chi_n - 1))
                
                return self.best_x, self.best_f

        cma = None
        last_best_f = np.inf
        stagnant_counter = 0
        
        while evals_used < self.budget:
            if time.time() - start_time > time_limit:
                break
                
            if cma is None:
                x0 = rng.uniform(-5, 5, self.dim)
                # Initial sigma: cover the box roughly
                sigma0 = 2.0
                cma = CMAES(x0, sigma0, popsize=9)
                
            bx, bf = cma.step(eval_func)
            
            if bf < best_f:
                best_f = bf
                best_x = bx
                stagnant_counter = 0
            else:
                stagnant_counter += 1
                
            # Restart if stagnant
            if stagnant_counter > 40:
                cma = None
                # Optional: If we are stuck, maybe try a larger sigma next time?
                # For now, just restart with random point.

        if best_x is None:
             # Fallback
             best_x = np.zeros(self.dim)
             best_f = func(best_x)
             
        return best_x, best_f
