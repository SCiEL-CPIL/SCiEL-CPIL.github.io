import numpy as np
import time

class Optimizer:
    """
    Single high-fidelity CMA-ES for BBOB f9 (Rosenbrock rotated).
    
    Strategy:
    1. Single CMA-ES instance: Avoids the budget waste of multi-starts. 
       Rosenbrock has a unique global minimum; a single large-sigma search 
       is more efficient at finding the basin than multiple small searches.
    2. Large initial sigma: Ensures the initial population covers the 
       [-5, 5]^5 box, allowing CMA-ES to locate the rotated valley.
    3. Standard CMA-ES: Uses the standard update rules with eigenvalue 
       clipping for stability. No separate local refinement phase, as 
       CMA-ES adapts its step size and covariance to the local structure.
    4. Budget Management: Strict time and call limits.
    """
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma0, max_calls, deadline, rng):
        """
        Run a single instance of CMA-ES.
        Returns (best_x, best_f, calls_made).
        """
        n = self.dim
        # CMA-ES parameters tuned for small dimensions
        # lambda_pop = 4 + 3 * ln(n) is standard
        lambda_pop = max(4, int(4 + 3 * np.log(n)))
        mu = lambda_pop // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        m = x0.copy()
        sigma = sigma0
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        calls_made = 0
        
        best_x = m.copy()
        best_f = func(m)
        calls_made += 1
        
        # Termination check
        if calls_made >= max_calls or time.time() >= deadline:
            return best_x, best_f, calls_made

        while calls_made < max_calls and time.time() < deadline:
            # Generate population
            ys = []
            fs = []
            for i in range(lambda_pop):
                z = rng.standard_normal(n)
                y = m + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                ys.append(y)
                fs.append(fy)
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    
                if calls_made >= max_calls:
                    break
            
            if len(ys) < lambda_pop:
                break
                
            ys = np.array(ys)
            fs = np.array(fs)
            
            # Sort
            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]
            
            # Update mean
            m_old = m.copy()
            m = sum(weights[i] * ys_sorted[i] for i in range(mu))
            
            # Update paths
            yinv = (ys_sorted[:mu] - m_old) / sigma
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (B @ (D * (m - m_old)))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (B @ (D * (m - m_old)))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance
            delta_d = (1 + 2 * max(0, mueff / n - 1)) * (np.linalg.norm(pc) ** 2 / n - 1)
            delta = (1 - c_1 - c_mu) * np.eye(n)
            delta += c_1 * (np.outer(pc, pc) + delta_d * np.eye(n))
            for i in range(mu):
                delta += c_mu * weights[i] * np.outer(yinv[i], yinv[i])
                
            # Eigen decomposition for stability
            eigvals, eigvecs = np.linalg.eigh(delta)
            eigvals = np.clip(eigvals, 1e-10, None)
            B = eigvecs
            D = np.sqrt(eigvals)
            
            # Termination
            if sigma < 1e-10:
                break
            if np.all(np.abs(ys_sorted[0] - ys_sorted[-1]) < 1e-8):
                break

        return best_x, best_f, calls_made

    def __call__(self, func):
        t_start = time.time()
        # Safety margin: 330s limit, keep 10s buffer
        deadline = t_start + 320
        
        n = self.dim
        total_budget = self.budget
        calls_made = 0
        
        # Single start at origin with large sigma
        # Origin is a safe bet as the minimum is in [-4, 4]^5
        # Large sigma (4.0) ensures the initial population covers the box
        x0 = np.zeros(n)
        sigma0 = 4.0
        
        rng = np.random.default_rng(42)
        
        x_best, f_best, c_used = self._cmaes(
            func, x0, sigma0, total_budget, deadline, rng
        )
        calls_made += c_used
        
        return x_best, f_best
