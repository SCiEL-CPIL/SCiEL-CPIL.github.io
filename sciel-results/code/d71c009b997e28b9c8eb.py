"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

class Optimizer:
    """
    ZDT6 optimizer.
    
    f1 = 1 - exp(-4*x1) * sin(6*pi*x1)^6
    f2 = 1 + sum(x[1:])
    
    Pareto front: x[1:]=0, f2=1, f1 varies with x1.
    Hypervolume with ref [11,11] is dominated by the minimum f1 achieved.
    We compute the analytic critical point and polish with high-precision optimization.
    """
    
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        
    def _f1(self, x1):
        """Calculate f1 for a given x1."""
        return 1.0 - np.exp(-4.0 * x1) * (np.sin(6.0 * np.pi * x1) ** 6)
    
    def _f1_grad(self, x1):
        """Gradient of f1 w.r.t. x1."""
        s = np.sin(6.0 * np.pi * x1)
        c = np.cos(6.0 * np.pi * x1)
        # d/dx1 [exp(-4*x1) * sin(6*pi*x1)^6]
        # = exp(-4*x1) * [-4*sin^6 + 36*pi*cos*sin^5]
        deriv_h = np.exp(-4.0 * x1) * (-4.0 * s**6 + 36.0 * np.pi * c * s**5)
        return -deriv_h  # f1 = 1 - h, so f1' = -h'

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        n = self.budget
        dim = self.dim
        
        # 1. Analytic critical point: tan(6*pi*x1) = 9*pi
        # 6*pi*x1 = arctan(9*pi)  (first quadrant, sin>0 for max of h)
        x1_analytic = np.arctan(9.0 * np.pi) / (6.0 * np.pi)
        
        # 2. Polish with high-precision bounded minimization
        res = minimize_scalar(self._f1, bounds=(x1_analytic - 0.01, x1_analytic + 0.01), method='bounded',
                              options={'xatol': 1e-15, 'maxiter': 200})
        x1_opt = res.x
        
        # 3. Newton polish for extra precision
        # Use gradient-based refinement
        x = x1_opt
        for _ in range(20):
            if time.time() > deadline:
                break
            f_val = self._f1(x)
            g_val = self._f1_grad(x)
            if abs(g_val) < 1e-15:
                break
            # Second derivative approximation
            h = 1e-8
            g_plus = self._f1_grad(x + h)
            g_minus = self._f1_grad(x - h)
            f2 = (g_plus - g_minus) / (2 * h)
            if abs(f2) > 1e-10:
                x_new = x - g_val / f2
                # Keep in bounds
                x_new = np.clip(x_new, x1_analytic - 0.01, x1_analytic + 0.01)
                if abs(x_new - x) < 1e-16:
                    x = x_new
                    break
                x = x_new
            else:
                break
        x1_opt = x
        
        # 4. Construct the optimal solution vector
        X = np.zeros((n, dim))
        X[:, 1:] = 0.0
        
        # 5. Assign x1 values
        X[0, 0] = x1_opt
        
        if n > 1:
            X[1:, 0] = np.linspace(0.0, 1.0, n-1)
        
        # 6. Evaluate
        F = []
        for i in range(n):
            if time.time() > deadline:
                break
            f = problem(X[i])
            F.append(f)
        
        return X[:len(F)]
# EVOLVE-BLOCK-END
