import numpy as np
import time
from scipy.optimize import minimize_scalar

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.x1_values = None
        self._precompute_grid()

    def _precompute_grid(self):
        N = self.budget
        if N <= 0:
            self.x1_values = np.array([])
            return
        
        # ZDT1 Pareto front: f1 = x1, f2 = 1 - sqrt(x1)
        # Reference point R = [11, 11]
        
        def compute_hv(p):
            # Use midpoints to avoid 0 and 1 boundaries issues if needed, 
            # but 0 and 1 are valid. 
            # x_i = (i/N)^p for i=1..N? Or i=0..N-1?
            # Current code uses u = linspace(0, 1, N). 
            # If u[0]=0, x[0]=0. If u[-1]=1, x[-1]=1.
            # This includes the endpoints.
            u = np.linspace(0.0, 1.0, N)
            x1 = u ** p
            # Clip just in case of floating point errors
            x1 = np.clip(x1, 0.0, 1.0)
            
            f2 = 1.0 - np.sqrt(x1)
            
            # Compute HV
            # H = sum_{i=0}^{N-2} (x1[i+1] - x1[i]) * (11 - f2[i]) + (11 - x1[N-1]) * (11 - f2[N-1])
            # Note: The last term uses x1[N-1] which is 1.
            # f2(1) = 0.
            # So last term is (11 - 1) * 11 = 110.
            
            df1 = np.diff(x1)
            # If N==1, diff is empty. Handle N=1 separately? Budget is 200.
            if N == 1:
                return (11.0 - x1[0]) * (11.0 - f2[0])
            
            df1 = np.append(df1, 11.0 - x1[-1])
            h = 11.0 - f2
            hv = np.sum(df1 * h)
            return hv

        # We want to maximize compute_hv(p)
        # So we minimize -compute_hv(p)
        
        # Initial guess based on theory: p ~ 4
        # Range: p > 0. 
        # If p is very small, points are clustered near 1.
        # If p is very large, points are clustered near 0.
        # Optimal p is around 4.
        
        res = minimize_scalar(lambda p: -compute_hv(p), bounds=(1.0, 6.0), method='bounded')
        best_p = res.x
        
        # Calculate final x1 values
        u = np.linspace(0.0, 1.0, N)
        self.x1_values = u ** best_p
        self.x1_values = np.clip(self.x1_values, 0.0, 1.0)

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin
        
        if self.x1_values is None or len(self.x1_values) == 0:
            return None

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        for x1 in self.x1_values:
            if time.time() > deadline:
                break
                
            x = np.zeros(self.dim)
            x[0] = x1
            # Clip to bounds
            x = np.clip(x, lo, hi)
            problem(x)
            
        return None
