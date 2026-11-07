import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 285.0  # Safety margin before 300s limit
        d = self.dim
        budget = self.budget
        
        def _time_ok():
            return time.time() - start_time < timeout

        # Track best globally
        best_x = None
        best_f = np.inf
        n_evals = 0

        def _update_best(x, f):
            nonlocal best_x, best_f, n_evals
            if f < best_f:
                best_f = f
                best_x = x.copy() if isinstance(x, np.ndarray) else np.array(x)

        # Phase 1: CMA-ES for global search with rotation handling
        # Parameters tuned for d=5
        lam = 8
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = 2 / ((d + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff)))
        
        x_mean = np.zeros(d)
        sigma = 2.0  # Start larger for better initial exploration in [-5,5]^5
        C = np.eye(d)
        B = np.eye(d)
        D = np.ones(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        it = 0
        chiN = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        # Use about 60% of budget for CMA-ES
        cma_budget = int(0.6 * budget)
        
        while n_evals + lam <= cma_budget and _time_ok():
            Z = np.random.randn(lam, d)
            Y = x_mean + sigma * (Z * D) @ B.T
            Y = np.clip(Y, -5, 5)
            
            F = np.empty(lam)
            for i in range(lam):
                F[i] = func(Y[i])
                n_evals += 1
                _update_best(Y[i], F[i])
            
            sorted_idx = np.argsort(F)
            Y_sorted = Y[sorted_idx]
            F_sorted = F[sorted_idx]
            
            y_old = x_mean.copy()
            x_mean = np.dot(w, Y_sorted[:mu])
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (B.T @ (D * (x_mean - y_old)))
            
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (it + 1)))) < (1 + 2 / (d + 1)) * cc
            it += 1
            
            if hsig:
                pc = (1 - cc) * pc + cc / sigma * (B.T @ (D * (x_mean - y_old)))
            
            delta_hebbian = c1 * (np.outer(ps, ps) + (1 - hsig) * (cc * (2 - cc) * np.outer(pc, pc)))
            
            for i in range(mu):
                delta_hebbian += (cmu * w[i]) * np.outer(Y_sorted[i] - y_old, Y_sorted[i] - y_old) / (sigma * sigma)
                
            C = (1 - c1 - cmu) * C + delta_hebbian
            C = (C + C.T) / 2
            
            D2, B = np.linalg.eigh(C)
            D = np.sqrt(np.clip(D2, 1e-20, None))
            
            sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            if sigma < 1e-6:
                break

        # Phase 2: Coordinate descent with bisection to exploit step-like structure
        # The f7 Step Ellipsoidal function has a plateau; we can use this
        # to refine the minimum by detecting when we're on the plateau
        if best_x is not None and n_evals < budget and _time_ok():
            remaining = budget - n_evals
            
            # First, do a few rounds of coordinate descent with adaptive step
            # to get close to the plateau boundary
            x = best_x.copy()
            f_best_local = best_f
            step = 0.5
            
            for round_num in range(10):
                if n_evals >= budget or not _time_ok():
                    break
                improved = False
                for axis in range(d):
                    if n_evals >= budget or not _time_ok():
                        break
                    # Try moving in both directions along this axis
                    for direction in [1, -1]:
                        x_try = x.copy()
                        x_try[axis] += direction * step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = func(x_try)
                        n_evals += 1
                        if f_try < f_best_local:
                            x = x_try.copy()
                            f_best_local = f_try
                            _update_best(x, f_try)
                            improved = True
                        else:
                            # If no improvement, shrink step for this axis
                            pass
                if not improved:
                    step *= 0.5
                    if step < 1e-8:
                        break
        
        # Phase 3: Nelder-Mead polish if budget remains
        if best_x is not None and n_evals < budget - 2 and _time_ok():
            remaining = budget - n_evals
            try:
                from scipy.optimize import minimize
                res = minimize(
                    func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': min(remaining, 200), 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except:
                pass

        # Final safety: if we haven't made any calls, make one
        if best_x is None:
            best_x = np.zeros(d)
            best_f = func(best_x)
            n_evals += 1

        return best_x, best_f
