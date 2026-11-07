import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        dim = self.dim
        calls = [0]
        budget = self.budget

        def safe_func(x):
            if calls[0] >= budget:
                return np.inf
            calls[0] += 1
            try:
                return float(func(np.asarray(x, dtype=float)))
            except Exception:
                return np.inf

        best_x = np.zeros(dim)
        best_f = np.inf

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)

        # --- Phase 1: Coarse Grid Search to find basin ---
        # Rastrigin has many local minima. A coarse grid helps identify the global basin.
        # We use a grid with step 2.0 in [-5, 5], so 6 points per dimension.
        # Total points: 6^5 = 7776. Too many for 500 budget.
        # Instead, we do a randomized coarse search or a smaller grid.
        # Let's try a grid with step 2.5 -> points: -5, -2.5, 0, 2.5, 5. (5 points)
        # 5^5 = 3125. Still too many.
        
        # Strategy: 
        # 1. Sample ~200 random points to get a rough idea of the landscape and find a good starting point.
        # 2. Use CMA-ES from the best point found.
        # 3. Use Nelder-Mead for final polish.
        
        # Since Rastrigin is multimodal, random restarts are key.
        # Let's split budget: 
        # - 100 calls for random sampling to find best initial point.
        # - 350 calls for CMA-ES.
        # - 50 calls for Nelder-Mead.
        
        # Phase 1: Random Sampling
        n_random = 100
        for _ in range(n_random):
            if calls[0] >= n_random:
                break
            x = np.random.uniform(-5, 5, dim)
            f = safe_func(x)
            record(x, f)
        
        # Phase 2: CMA-ES
        # We will run CMA-ES from the best point found so far.
        # If CMA-ES stagnates, we might restart from another good point.
        
        def run_cma_es(x0, max_calls, max_time):
            """
            Runs CMA-ES starting from x0, using up to max_calls and max_time.
            Returns best_x, best_f found during this run.
            """
            if calls[0] >= max_calls or time.time() > max_time:
                return best_x, best_f

            # CMA-ES Parameters
            lambda_ = 10
            mu = 5
            w = np.log(mu / 2 + 1) - np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            mu_eff = 1.0 / np.sum(w**2)
            
            m = np.array(x0, dtype=float)
            sigma = 1.0 # Start with smaller sigma for local search
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
            
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)
                
            local_best_x = m.copy()
            local_best_f = safe_func(m)
            record(m, local_best_f)
            
            # We will use the global best_x, best_f but update them via record
            # We need to track local best for CMA-ES decisions
            
            for gen in range(100):
                if calls[0] >= max_calls or time.time() > max_time:
                    break
                    
                xs = []
                fs = []
                for k in range(lambda_):
                    z = np.random.randn(dim)
                    x = m + sigma * (L @ z)
                    xs.append(x)
                    fs.append(safe_func(x))
                
                idx = np.argsort(fs)
                xs = [xs[i] for i in idx]
                fs = [fs[i] for i in idx]
                
                for x, f in zip(xs, fs):
                    record(x, f)
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = x.copy()
                
                m_new = sum(w[i] * xs[i] for i in range(mu))
                sigma_new = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))
                
                v = [(xs[i] - m) / sigma for i in range(mu)]
                C_mu = sum(w[i] * np.outer(v[i], v[i]) for i in range(mu))
                pc_new = (1 - c_c) * pc + c_c * ((m_new - m) / sigma)
                C_1 = (1 - (c_1 + c_mu)) * C + c_1 * (1 + (1 - c_sigma) * d_sigma * np.outer(pc, pc) - c_sigma * np.outer(pc, pc)) + c_mu * (C_mu - C)
                
                C = (C_1 + C_1.T) / 2
                C += 1e-10 * np.eye(dim)
                
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * ((m_new - m) / sigma)
                
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = C + 1e-3 * np.eye(dim)
                    L = np.linalg.cholesky(C)
                    
                m = m_new
                sigma = sigma_new
                pc = pc_new
                
                if sigma < 1e-5:
                    break

            return local_best_x, local_best_f

        # Run CMA-ES from the current best point
        # Reserve 50 calls for NM
        cma_limit = budget - 50
        cma_deadline = deadline - 10
        
        # It might be beneficial to restart CMA-ES if it gets stuck.
        # However, with a small budget, a single robust run from a good start is often better.
        # Let's try a few short CMA-ES runs from the top 3 random points found.
        
        # We need to store the top random points
        # Since record only keeps best, let's re-implement a small pool.
        
        # Actually, let's just run CMA-ES from the current best.
        # If we have time/calls left, we can try NM.
        
        run_cma_es(best_x, cma_limit, cma_deadline)

        # Phase 3: Nelder-Mead Polish
        if calls[0] < budget - 5:
            self._nelder_mead(safe_func, best_x, calls, max_iters=30, step0=0.3)

        return best_x, best_f

    def _nelder_mead(self, func, x0, calls, max_iters=30, step0=0.3):
        """Nelder-Mead for final polish."""
        d = self.dim
        x = np.array(x0, dtype=float)
        f0 = func(x)
        simplex = [x.copy()]
        fs = [f0]
        for i in range(d):
            xi = x.copy()
            xi[i] += step0
            simplex.append(xi)
            fs.append(func(xi))
        best_x = x.copy()
        best_f = f0
        
        for _ in range(max_iters):
            if calls[0] >= self.budget:
                break
            order = np.argsort(fs)
            simplex = [simplex[i] for i in order]
            fs = [fs[i] for i in order]
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = simplex[0].copy()
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + (centroid - simplex[-1])
            fr = func(xr)
            if fr < fs[0]:
                # Extension
                xe = centroid + 2.0 * (centroid - simplex[-1])
                fe = func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fs[-1] = fe
                else:
                    simplex[-1] = xr
                    fs[-1] = fr
                continue
            if fr < fs[-2]:
                simplex[-1] = xr
                fs[-1] = fr
                continue
            
            # Contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = func(xc)
            if fc < fs[-1]:
                simplex[-1] = xc
                fs[-1] = fc
                continue
            
            # Shrink
            for i in range(1, d + 1):
                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                fs[i] = func(simplex[i])
                
        return best_x, best_f
