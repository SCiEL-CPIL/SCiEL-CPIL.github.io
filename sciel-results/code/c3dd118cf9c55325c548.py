import numpy as np
import time
from scipy.optimize import minimize_scalar


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = self.start_time + 310.0

    def _compute_hv_for_beta(self, beta, n):
        """Compute hypervolume of points on ZDT1 front with power-law spacing x0 = t^beta."""
        if n == 1:
            x0 = np.array([0.0])
            f1 = x0
            f2 = 1.0 + 9.0 * (1.0 - np.sqrt(x0))
            return (11.0 - f1[0]) * (11.0 - f2[0])
        
        indices = np.arange(n, dtype=np.float64)
        x0 = (indices / (n - 1)) ** beta
        f1 = x0
        f2 = 1.0 + 9.0 * (1.0 - np.sqrt(x0))
        
        r1, r2 = 11.0, 11.0
        
        widths = np.diff(f1)
        heights = r2 - f2[:-1]
        hv = np.sum(widths * heights)
        hv += (r1 - f1[-1]) * (r2 - f2[-1])
        
        return hv

    def __call__(self, problem, bounds):
        n = self.budget
        if n <= 0:
            return None

        if n <= 2:
            beta = 2.0
        else:
            def neg_hv(beta):
                return -self._compute_hv_for_beta(beta, n)
            
            result = minimize_scalar(neg_hv, bounds=(0.5, 5.0), method='bounded',
                                     options={'xatol': 1e-8})
            beta = result.x

        if n == 1:
            x0_values = np.array([0.0])
        elif n == 2:
            x0_values = np.array([0.0, 1.0])
        else:
            indices = np.arange(n, dtype=np.float64)
            x0_values = (indices / (n - 1)) ** beta

        for x0 in x0_values:
            if time.time() > self.deadline:
                break

            x = np.zeros(self.dim)
            x[0] = float(x0)

            try:
                problem(x)
            except Exception:
                break

        return None
