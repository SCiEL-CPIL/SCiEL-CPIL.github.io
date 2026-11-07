import time
import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        dim = self.dim
        lo, hi = -5.0, 5.0
        rng = np.random.default_rng(42)
        
        calls = [0]
        best_x = None
        best_f = np.inf

        def counted(x):
            nonlocal best_f, best_x
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            calls[0] += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def cma_es(x0, budget_remaining):
            lambda_ = 12 
            mu = lambda_ // 2 
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            sigma = 2.0 
            x_m = np.array(x0, dtype=float)
            C = np.eye(dim)
            
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            while calls[0] < budget_remaining and time.time() < deadline:
                # Sample
                z = rng.standard_normal((lambda_, dim))
                
                # Update sampling matrix
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_sqrt = eigvecs * np.sqrt(eigvals)
                
                pops = x_m + sigma * (z @ C_sqrt.T)
                
                # Evaluate
                fvals = np.array([counted(pops[i]) for i in range(lambda_)])
                
                # Select
                order = np.argsort(fvals)
                y_sel = z[order[:mu]]
                
                # Update mean
                x_m = x_m + sigma * (weights @ y_sel)
                
                # Update covariance
                p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * (weights @ y_sel)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt @ (weights @ y_sel))
                
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(p_c, p_c) + (1 - (1 - d_sigma) * c_sigma) * np.eye(dim))
                for i in range(mu):
                    C += c_mu * weights[i] * np.outer(y_sel[i], y_sel[i])
                    
                C = (C + C.T) / 2
                
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) - np.sqrt(dim)))
                
                if sigma < 1e-8:
                    break

            return x_m

        # Phase 1: CMA-ES with warm restarts
        # Split budget into 3 chunks
        chunks = 3
        budget_per_chunk = self.budget // chunks
        
        current_start = np.zeros(dim)
        for i in range(chunks):
            if calls[0] >= self.budget or time.time() >= deadline:
                break
            current_start = cma_es(current_start, budget_per_chunk)
            # The next start is the best point found so far (which is updated in counted)
            if best_x is not None:
                current_start = best_x.copy()
            
        # Phase 2: Nelder-Mead Polish
        if calls[0] < self.budget and time.time() < deadline and best_x is not None:
            x = best_x.copy()
            f0 = counted(x)
            simplex = np.empty((dim + 1, dim))
            simplex[0] = x
            fvals = np.empty(dim + 1)
            fvals[0] = f0
            
            for i in range(dim):
                p = x.copy()
                p[i] += 0.5
                p = np.clip(p, lo, hi)
                simplex[i + 1] = p
                fvals[i + 1] = counted(p)
                
            alpha, gamma, rho = 1.0, 2.0, 0.5
            while calls[0] < self.budget and time.time() < deadline:
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                x_best = simplex[0]
                x_worst = simplex[-1]
                centroid = simplex[:-1].mean(axis=0)
                
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, lo, hi)
                f_r = counted(x_r)
                
                if f_r < fvals[0]:
                    x_e = centroid + gamma * (centroid - x_worst)
                    x_e = np.clip(x_e, lo, hi)
                    f_e = counted(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    x_c = centroid + rho * (x_worst - centroid)
                    x_c = np.clip(x_c, lo, hi)
                    f_c = counted(x_c)
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
                            fvals[i] = counted(simplex[i])
                            
                if np.std(fvals) < 1e-10:
                    break

        return best_x, best_f
