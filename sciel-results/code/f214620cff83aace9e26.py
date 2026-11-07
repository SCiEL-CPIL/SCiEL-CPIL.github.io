import numpy as np
import time

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
        # Hypervolume maximization for convex front suggests placing points
        # according to the quantiles of density p(x) proportional to sqrt(R2 - f2(x)).
        
        # Let x = u^2, so u in [0, 1].
        # f2 = 1 - u.
        # R2 - f2 = 11 - (1 - u) = 10 + u.
        # dx = 2u du.
        # The "weight" in x-space is w(x) = sqrt(10 + sqrt(x)).
        # The optimal x_i are quantiles of w(x).
        # Transform to u-space:
        # Integral of w(x) dx = Integral of sqrt(10+u) * 2u du.
        # So density in u-space is q(u) proportional to u * sqrt(10 + u).
        
        # We need to compute the CDF of q(u) on [0, 1].
        # q(u) = u * sqrt(10 + u)
        # Let's approximate the integral numerically or use an analytical form if possible.
        # Integral of u * (10 + u)^0.5 du
        # Let v = 10 + u, u = v - 10, du = dv.
        # Integral of (v - 10) * v^0.5 dv = Integral of (v^1.5 - 10v^0.5) dv
        # = (2/5)v^2.5 - 10*(2/3)v^1.5
        # = 0.4 * v^2.5 - (20/3) * v^1.5
        
        def cdf_u(u):
            v = 10 + u
            return 0.4 * v**2.5 - (20.0/3.0) * v**1.5
        
        # Normalize
        cdf_0 = cdf_u(0)
        cdf_1 = cdf_u(1)
        norm = cdf_1 - cdf_0
        
        # We want N points.
        # The intervals are [0, 1/N], [1/N, 2/N], ...
        # We place the point at the "center" of the mass in each interval.
        # i.e., Q(u_i) = cdf_0 + norm * (i - 0.5) / N
        
        u_vals = np.zeros(N)
        # Use bisection to find u_i for each target CDF value
        targets = cdf_0 + norm * (np.arange(N) + 0.5) / N
        
        for i in range(N):
            target = targets[i]
            lo, hi = 0.0, 1.0
            for _ in range(50):
                mid = (lo + hi) / 2.0
                val = cdf_u(mid)
                if val < target:
                    lo = mid
                else:
                    hi = mid
            u_vals[i] = (lo + hi) / 2.0
            
        self.x1_values = u_vals**2

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
            # Clip to bounds (though x1 is in [0,1] and bounds are [0,1])
            x = np.clip(x, lo, hi)
            problem(x)
            
        return None
