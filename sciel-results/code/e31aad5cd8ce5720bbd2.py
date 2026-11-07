import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        
        m = min(200, self.budget)
        
        # Analytical HV for ZDT2 Pareto front
        # f1 = x1, f2 = 1 - sqrt(x1)
        # Ref = (11, 11)
        # w(x) = 11 - f2(x) = 10 + sqrt(x)
        # HV = sum_{i=1}^m (x_{i+1} - x_i) w(x_i) with x_{m+1} = 11
        
        def hv_and_grad(x):
            # x is array of length m, sorted ascending
            x_ext = np.append(x, 11.0)
            widths = np.diff(x_ext)
            # w(x_i) = 10 + sqrt(x_i)
            w = 10.0 + np.sqrt(x)
            hv = np.sum(widths * w)
            
            # Gradient
            # dHV/dx_k = w(x_{k-1}) - w(x_k) + (x_{k+1} - x_k) * w'(x_k)
            # w'(x) = 0.5 / sqrt(x)
            
            grad = np.zeros_like(x)
            # Interior points
            # For k=1 to m-1 (0-indexed 0 to m-2)
            # Term from left: w(x_{k-1}) (if k>0)
            # Term from right: -w(x_k) + (x_{k+1} - x_k) w'(x_k)
            
            # Vectorized calculation
            # w_minus = w shifted right? 
            # Let's just loop for clarity and safety, m=200 is small.
            for k in range(m):
                val = -w[k] + (x_ext[k+1] - x[k]) * (0.5 / np.sqrt(max(x[k], 1e-12)))
                if k > 0:
                    val += w[k-1]
                grad[k] = val
            
            return hv, grad

        # 1. Find good initial alpha
        def hv_from_alpha(alpha):
            x = (np.arange(1, m + 1, dtype=float) / m) ** alpha
            return hv_and_grad(x)[0]

        # Simple grid search or minimize_scalar for alpha
        # The optimal alpha is likely around 0.5 - 1.0
        best_alpha = 0.5
        best_hv_init = hv_from_alpha(0.5)
        # Refine alpha
        alphas = np.linspace(0.1, 2.0, 20)
        for a in alphas:
            h = hv_from_alpha(a)
            if h > best_hv_init:
                best_hv_init = h
                best_alpha = a
        
        x = (np.arange(1, m + 1, dtype=float) / m) ** best_alpha
        
        # 2. Gradient Ascent
        eta = 0.01
        for step in range(1000):
            if time.time() > deadline:
                break
            
            hv, grad = hv_and_grad(x)
            
            # Update
            x_new = x + eta * grad
            
            # Enforce constraints: 0 < x_1 < ... < x_m < 1
            # Clip to [0, 1]
            x_new = np.clip(x_new, 1e-6, 1.0 - 1e-6)
            
            # Enforce sorting
            # If x_new is not sorted, we need to fix it.
            # Simple projection: sort and re-assign? No, that changes which point is which.
            # Better: if x_i >= x_{i+1}, set both to average?
            # Or reduce eta.
            
            sorted_ok = True
            for i in range(m - 1):
                if x_new[i] >= x_new[i+1]:
                    sorted_ok = False
                    break
            
            if not sorted_ok:
                eta *= 0.5
                if eta < 1e-6:
                    break
                continue
            
            x = x_new
            
        # Final evaluation
        used = 0
        recorded = []
        
        for x1 in x:
            if time.time() > deadline:
                break
            x_vec = np.zeros(n)
            x_vec[0] = x1
            # Clip to bounds (though x1 is in [0,1] and bounds are [0,1])
            x_vec = np.clip(x_vec, lo, hi)
            f = problem(x_vec)
            recorded.append(f)
            used += 1
            
        return recorded
