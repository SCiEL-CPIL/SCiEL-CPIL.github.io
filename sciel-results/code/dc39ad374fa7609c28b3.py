import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Leave 10 seconds margin for safety (300s limit)
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline

        # CMA-ES Parameters
        lam = 10
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (20 * dim ** 2))
        
        # State variables
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        best_x = x.copy()
        best_f = func(x)
        calls = 1
        
        last_improvement_call = 0
        stagnation_limit = 150
        
        # Main CMA-ES loop
        while calls < budget and time_ok():
            y_list = []
            f_list = []
            
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.clip(eigenvalues, 1e-12, None)
            D = np.sqrt(eigenvalues)
            B = eigenvectors
            
            for i in range(lam):
                if not time_ok() or calls >= budget:
                    break
                z = np.random.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                y_list.append(y)
                f_list.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    last_improvement_call = calls
            
            if len(y_list) < mu:
                break
                
            idx = np.argsort(f_list)
            y_selected = np.array([y_list[i] for i in idx[:mu]])
            
            m = np.sum(weights[:, None] * y_selected, axis=0)
            
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * (m - x)))
            
            gen_count = max(1, calls // lam)
            h_sigma = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * gen_count)) < (1.4 + 2 / (dim + 1)) * chi_n
            
            if h_sigma:
                pc = (1 - c_c) * pc + c_c * (m - x) / sigma
                delta_h = 1 - h_sigma
            else:
                pc = (1 - c_c) * pc
                delta_h = 1
            
            C = (1 - c_c * (2 - c_c + delta_h)) * C + c_c * (2 - c_c + delta_h) * (pc @ pc.T)
            C = (C + C.T) / 2.0
            
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            x = m
            
            if sigma > 5.0:
                sigma = 5.0
            if sigma < 1e-4:
                break

            if calls - last_improvement_call > stagnation_limit and calls < budget * 0.8:
                x = best_x + np.random.normal(0, 0.5, dim)
                x = np.clip(x, -5, 5)
                sigma = 1.0
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                last_improvement_call = calls

        # Phase 2: Nelder-Mead local refinement on the best point found
        nm_budget = budget - calls
        if nm_budget > dim + 1 and time_ok():
            best_x, best_f, calls = self._nelder_mead(func, best_x, nm_budget, dim, deadline, calls)
        
        return best_x, best_f

    def _nelder_mead(self, func, x0, budget, dim, deadline, initial_calls):
        """Nelder-Mead local optimizer starting from x0."""
        def time_ok():
            return time.time() < deadline
        
        best_f = func(x0)
        best_x = x0.copy()
        calls = initial_calls + 1
        
        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrinkage
        
        # Initialize simplex with small perturbations
        simplex = [x0.copy() for _ in range(dim + 1)]
        fvals = [best_f]
        
        # Create initial simplex
        for i in range(dim):
            simplex[i] = x0.copy()
            simplex[i][i] += 0.1
            simplex[i] = np.clip(simplex[i], -5, 5)
            fvals.append(func(simplex[i]))
            calls += 1
            if fvals[-1] < best_f:
                best_f = fvals[-1]
                best_x = simplex[i].copy()
        
        max_iter = budget - calls
        for iteration in range(max_iter):
            if calls >= budget or not time_ok():
                break
            
            # Sort simplex by function values
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Compute centroid of all but worst point
            centroid = np.mean(simplex[:dim], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[dim])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            if fr < best_f:
                best_f = fr
                best_x = xr.copy()
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[dim])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                if fe < best_f:
                    best_f = fe
                    best_x = xe.copy()
                
                if fe < fr:
                    simplex[dim] = xe
                    fvals[dim] = fe
                else:
                    simplex[dim] = xr
                    fvals[dim] = fr
            elif fr < fvals[dim - 1]:
                # Accept reflection
                simplex[dim] = xr
                fvals[dim] = fr
            else:
                # Contraction
                if fr < fvals[dim]:
                    # Outside contraction
                    xc = centroid + rho * (simplex[dim] - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[dim])
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                if fc < best_f:
                    best_f = fc
                    best_x = xc.copy()
                
                if fc < min(fr, fvals[dim]):
                    simplex[dim] = xc
                    fvals[dim] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if fvals[i] < best_f:
                            best_f = fvals[i]
                            best_x = simplex[i].copy()
                    
                    if calls >= budget:
                        break
        
        return best_x, best_f, calls
