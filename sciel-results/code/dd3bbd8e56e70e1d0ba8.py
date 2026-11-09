import numpy as np
import time
import scipy.stats.qmc
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls_made = 0
        self.start_time = None
        self.deadline = None

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 280.0
        self.calls_made = 0
        rng = np.random.default_rng(42)
        
        def wrapped_func(x):
            self.calls_made += 1
            if time.time() > self.deadline - 5.0:
                raise BudgetExhausted()
            if self.calls_made > self.budget:
                raise BudgetExhausted()
            return func(x)

        class BudgetExhausted(Exception):
            pass

        # 1. Global Scan
        n_scan = 40
        sampler = scipy.stats.qmc.Sobol(d=self.dim, scramble=True, seed=42)
        points = sampler.random(n_scan)
        points = points * 10 - 5
        
        best_x = None
        best_f = np.inf
        candidates = []
        
        for x in points:
            try:
                fx = wrapped_func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                candidates.append((fx, x.copy()))
            except BudgetExhausted:
                break

        if best_x is None:
            return np.zeros(self.dim), 0.0

        # 2. Local Refinement using Nelder-Mead on top candidates
        candidates.sort(key=lambda t: t[0])
        top_candidates = candidates[:3]
        
        remaining_budget = self.budget - self.calls_made
        
        # Allocate budget for refinement
        # Nelder-Mead uses dim+1 points per iteration.
        # Let's try to run 10-15 iterations for the best, less for others.
        
        for i, (f_c, x_c) in enumerate(top_candidates):
            if self.calls_made >= self.budget or time.time() > self.deadline - 10.0:
                break
            
            # Determine max iterations based on remaining budget and candidate rank
            # Best candidate gets more budget
            if i == 0:
                max_iter = 20
            elif i == 1:
                max_iter = 10
            else:
                max_iter = 5
                
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x_c, 
                    method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except BudgetExhausted:
                pass
            except Exception:
                pass

        # 3. Final Polish with small random steps if budget remains
        if self.calls_made < self.budget and time.time() < self.deadline - 5.0:
            polish_sigma = 0.1
            for _ in range(50): # Max 50 polish steps
                if self.calls_made >= self.budget or time.time() > self.deadline - 5.0:
                    break
                y = np.clip(best_x + polish_sigma * rng.standard_normal(self.dim), -5, 5)
                try:
                    fy = wrapped_func(y)
                    if fy < best_f:
                        best_f = fy
                        best_x = y
                        polish_sigma *= 0.95
                    else:
                        polish_sigma *= 0.95
                except BudgetExhausted:
                    break

        return best_x, best_f
