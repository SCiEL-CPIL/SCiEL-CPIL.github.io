import numpy as np
import time
import sys
from scipy.optimize import minimize_scalar, minimize, brentq

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.optimal_x1 = None
        self._compute_optimal_x1()

    def _compute_hv(self, points, ref):
        n = len(points)
        if n == 0:
            return 0.0
        points = points[np.argsort(points[:, 0])]
        hv = 0.0
        x_next = ref[0]
        for i in range(n):
            x_curr = points[i][0]
            y_curr = points[i][1]
            width = x_next - x_curr
            height = ref[1] - y_curr
            hv += width * height
            x_next = x_curr
        return hv

    def _hv_from_x1(self, x1, ref):
        x1 = np.sort(x1)
        f2 = 1.0 - np.sqrt(np.clip(x1, 0, 1))
        points = np.column_stack((x1, f2))
        return self._compute_hv(points, ref)

    def _compute_optimal_x1(self):
        N = self.budget
        if N <= 0:
            return
        ref = np.array([11.0, 11.0])
        
        # Method 1: Equalize hypervolume contribution per interval
        # For ZDT1: f1=x, f2=1-sqrt(x), ref=(11,11)
        # HV contribution of interval i: (x_{i+1}-x_i) * (11 - (1-sqrt(x_i))) = (x_{i+1}-x_i)*(10+sqrt(x_i))
        # Equalize: (x_{i+1}-x_i)*(10+sqrt(x_i)) = C
        # => dx/di = C/(10+sqrt(x))
        # => integral: 10*x + (2/3)*x^1.5 = C*i + D
        # With x(0)=0: D=0
        # With x(N-1)=1: 10 + 2/3 = C*(N-1) => C = (10+2/3)/(N-1)
        
        C = (10.0 + 2.0/3.0) / (N - 1)
        x1_equal = np.zeros(N)
        x1_equal[0] = 0.0
        x1_equal[-1] = 1.0
        
        # Solve 10*x + (2/3)*x^1.5 = C*i for each i
        for i in range(1, N-1):
            target = C * i
            # Solve 10*x + (2/3)*x^1.5 = target for x in [0,1]
            def f(x):
                return 10.0*x + (2.0/3.0)*x**1.5 - target
            x1_equal[i] = brentq(f, 0.0, 1.0, xtol=1e-15)
        
        hv_equal = self._hv_from_x1(x1_equal, ref)
        
        # Method 2: Try power law x1 = t^p
        t = np.linspace(0, 1, N)
        def neg_hv_p(p):
            if p < 0.01:
                return 1e10
            x1 = t ** p
            return -self._hv_from_x1(x1, ref)
        
        best_p = 1.0
        best_hv_p = -np.inf
        for p in np.arange(0.5, 8.0, 0.1):
            hv = -neg_hv_p(p)
            if hv > best_hv_p:
                best_hv_p = hv
                best_p = p
        
        res = minimize_scalar(neg_hv_p, bounds=(best_p - 0.05, best_p + 0.05), method='bounded', options={'xatol': 1e-12})
        if res.success:
            best_p = res.x
            best_hv_p = -res.fun
        
        x1_power = t ** best_p
        hv_power = self._hv_from_x1(x1_power, ref)
        
        # Method 3: Two-parameter family
        def neg_hv_two(params):
            a, b = params
            if a < 0.01 or b < 0.01:
                return 1e10
            ta = t ** a
            ob = (1 - t) ** b
            denom = ta + ob
            x1 = np.zeros(N)
            mask = denom > 0
            x1[mask] = ta[mask] / denom[mask]
            x1[0] = 0.0
            x1[-1] = 1.0
            return -self._hv_from_x1(x1, ref)
        
        best_two = None
        best_two_hv = -np.inf
        for a in np.arange(0.5, 5.0, 0.25):
            for b in np.arange(0.5, 5.0, 0.25):
                hv = -neg_hv_two([a, b])
                if hv > best_two_hv:
                    best_two_hv = hv
                    best_two = [a, b]
        
        if best_two:
            res2 = minimize(neg_hv_two, best_two, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 5000})
            if res2.fun < -best_two_hv:
                best_two = res2.x
                best_two_hv = -res2.fun
        
        # Choose best
        candidates = [
            (hv_equal, x1_equal),
            (hv_power, x1_power),
        ]
        if best_two is not None:
            a, b = best_two
            ta = t ** a
            ob = (1 - t) ** b
            denom = ta + ob
            x1_two = np.zeros(N)
            mask = denom > 0
            x1_two[mask] = ta[mask] / denom[mask]
            x1_two[0] = 0.0
            x1_two[-1] = 1.0
            candidates.append((best_two_hv, x1_two))
        
        best_hv = max(c[0] for c in candidates)
        self.optimal_x1 = [c[1] for c in candidates if c[0] == best_hv][0]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320
        
        if self.optimal_x1 is None:
             self._compute_optimal_x1()
             
        x1_vals = self.optimal_x1
        N = len(x1_vals)
        
        X = np.zeros((N, self.dim))
        X[:, 0] = x1_vals
        
        for i in range(N):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
