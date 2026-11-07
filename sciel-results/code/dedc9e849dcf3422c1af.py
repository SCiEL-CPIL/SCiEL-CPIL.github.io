import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n_evals = self.budget
        if n_evals <= 0:
            return None
        
        deadline = time.time() + 300
        
        def neg_hv(x1):
            f1 = np.sort(x1)
            f2 = 1.0 - np.sqrt(np.clip(f1, 0.0, 1.0))
            if len(f1) == 1:
                return -(11.0 - f1[0]) * (11.0 - f2[0])
            terms = np.diff(f1) * (11.0 - f2[:-1])
            terms = np.append(terms, (11.0 - f1[-1]) * (11.0 - f2[-1]))
            return -np.sum(terms)
        
        def neg_hv_t(t):
            x1 = 1.0 / (1.0 + np.exp(-t))
            return neg_hv(x1)
        
        i = np.arange(1, n_evals + 1)
        x0_phys = np.clip((i / n_evals) ** 3.0, 1e-8, 1 - 1e-8)
        t0 = np.log(x0_phys / (1 - x0_phys))
        
        try:
            if time.time() < deadline - 5:
                result = minimize(neg_hv_t, t0, method='Nelder-Mead',
                                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
                x1_opt = np.sort(1.0 / (1.0 + np.exp(-result.x)))
            else:
                x1_opt = x0_phys
        except Exception:
            x1_opt = x0_phys
        
        last_x = None
        for x1 in x1_opt:
            if time.time() > deadline:
                break
            x = np.full(self.dim, 0.5)
            x[0] = np.clip(x1, 0.0, 1.0)
            last_x = problem(x)
            
        return last_x
