import numpy as np
import time

class Optimizer:
    """
    Optimizer using CMA-ES followed by Nelder-Mead Polish.
    CMA-ES adapts to the anisotropic landscape of the sharp ridge.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = time.time()
        self.deadline = 290.0

    def _call_func(self, x):
        if self.calls >= self.budget or time.time() > self.deadline:
            return np.inf
        self.calls += 1
        try:
            val = self.func(x)
            return val if np.isfinite(val) else np.inf
        except Exception:
            return np.inf

    def _cma_es(self, x0, f0, budget_limit):
        dim = self.dim
        # Population size
        lam = 4 + int(3 * np.log(dim))
        if lam > 10: lam = 10 # Cap population size to save budget
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial state
        m = x0.copy()
        sigma = 2.0 # Initial step size
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Precompute
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        best_x = x0
        best_f = f0
        
        # Main loop
        while self.calls < budget_limit and time.time() < self.deadline:
            # Sample population
            ys = np.random.multivariate_normal(np.zeros(dim), C, size=lam)
            xs = m + sigma * ys
            
            # Evaluate
            fs = np.array([self._call_func(x) for x in xs])
            
            # Update best
            idx_min = np.argmin(fs)
            if fs[idx_min] < best_f:
                best_f = fs[idx_min]
                best_x = xs[idx_min].copy()
                
            # Sort
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]
            
            # Update mean
            m_new = m + sigma * np.sum(weights[:, None] * ys[:mu], axis=0)
            
            # Update covariance
            # Rank-mu update
            y_diff = ys[:mu] - np.sum(weights[:, None] * ys[:mu], axis=0)
            C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (np.linalg.norm(pc) > (1 + 0.5 * c_c) * chi_n) * (chi_n**2 - 1) * np.eye(dim)) + c_mu * np.sum(weights[:, None, None] * (y_diff[:, :, None] * y_diff[:, None, :]), axis=0)
            
            # Ensure C is symmetric positive definite (approx)
            C = (C + C.T) / 2
            
            # Update path evolution
            pc = (1 - c_c) * pc + c_c * np.sign(m_new - m) / sigma * chi_n
            ps = np.sqrt(1 - (1 - c_sigma)**2) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (m_new - m)
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update mean
            m = m_new
            
            # Check convergence
            if sigma < 1e-10:
                break
            # Check if spread is small
            if np.all(np.abs(ys[:mu]) < 1e-6):
                break
                
        return best_x, best_f

    def _nelder_mead(self, x0, f0, budget_limit):
        dim = self.dim
        x = x0.copy()
        f = f0
        
        # Create initial simplex
        simplex = [x]
        simplex_f = [f]
        
        step = 0.1
        for i in range(dim - 1):
            x_pert = x.copy()
            x_pert[i] += step
            f_pert = self._call_func(x_pert)
            simplex.append(x_pert)
            simplex_f.append(f_pert)
            
        simp = sorted(zip(simplex_f, simplex), key=lambda t: t[0])
        
        while self.calls < budget_limit and time.time() < self.deadline:
            if len(simp) < dim + 1:
                break
                
            f_worst, x_worst = simp[-1]
            f_best, x_best = simp[0]
            f_second, x_second = simp[1]
            
            centroid = np.sum([x for _, x in simp[:-1]], axis=0) / dim
            
            # Reflection
            alpha = 1.0
            x_ref = centroid + alpha * (centroid - x_worst)
            f_ref = self._call_func(x_ref)
            
            if f_ref < f_second:
                simp[-1] = (f_ref, x_ref)
                simp.sort(key=lambda t: t[0])
            else:
                gamma = 2.0
                x_exp = centroid + gamma * (centroid - x_worst)
                f_exp = self._call_func(x_exp)
                
                if f_exp < f_ref:
                    simp[-1] = (f_exp, x_exp)
                    simp.sort(key=lambda t: t[0])
                else:
                    rho = 0.5
                    x_con = centroid + rho * (x_worst - centroid)
                    f_con = self._call_func(x_con)
                    
                    if f_con < f_worst:
                        simp[-1] = (f_con, x_con)
                        simp.sort(key=lambda t: t[0])
                    else:
                        for i in range(1, len(simp)):
                            x_shrink = x_best + 0.5 * (simp[i][1] - x_best)
                            f_shrink = self._call_func(x_shrink)
                            simp[i] = (f_shrink, x_shrink)
                        simp.sort(key=lambda t: t[0])
                        
                        spread = np.max([np.linalg.norm(x_best - x) for _, x in simp])
                        if spread < 1e-7:
                            break
        
        return simp[0][1], simp[0][0]

    def __call__(self, func):
        self.func = func
        self.calls = 0
        self.start_time = time.time()
        
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-5, 5, self.dim)
        f0 = self._call_func(x0)
        
        # Budget split: 70% CMA-ES, 30% Nelder-Mead
        budget_cma = int(self.budget * 0.7)
        budget_nm = self.budget - budget_cma
        
        x_cma, f_cma = self._cma_es(x0, f0, budget_cma)
        x_nm, f_nm = self._nelder_mead(x_cma, f_cma, budget_nm)
        
        if f_nm < f_cma:
            return x_nm, f_nm
        else:
            return x_cma, f_cma
