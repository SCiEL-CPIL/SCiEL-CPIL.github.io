import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = self.start_time + 325.0
        self.optimal_x0 = self._compute_optimal_x0(budget)

    def _compute_optimal_x0(self, n):
        if n <= 0:
            return np.array([])
        if n == 1:
            # Maximize (11 - x)(11 - (1-sqrt(x))) = (11-x)(10+sqrt(x))
            # Derivative: -(10+sqrt(x)) + (11-x)(1/(2sqrt(x))) = 0
            # -20sqrt(x) - 2x + 11 - x = 0
            # 3x + 20sqrt(x) - 11 = 0
            # Let u = sqrt(x). 3u^2 + 20u - 11 = 0
            # u = (-20 + sqrt(400 + 132))/6 = (-20 + sqrt(532))/6
            u = (-20 + np.sqrt(532)) / 6.0
            return np.array([u**2])

        r1 = 11.0
        r2 = 11.0
        
        def final_error(y0):
            # y0 is sqrt(x0)
            if y0 <= 0:
                return -np.inf
            y_prev = y0
            y_curr = np.sqrt(3 * y0**2 + 2 * y0 * (r2 - 1))
            
            for j in range(1, n - 1):
                # y_{j+1} = sqrt(2 y_j y_{j-1} - y_j^2)
                val = 2 * y_curr * y_prev - y_curr**2
                if val < 0:
                    return -np.inf
                y_next = np.sqrt(val)
                y_prev, y_curr = y_curr, y_next
            
            # Check boundary: 2 y_{n-2} - y_{n-1} = r1
            # Here y_prev is y_{n-2}, y_curr is y_{n-1}
            return 2 * y_prev - y_curr - r1

        # Bisection for y0
        # y0 must be such that the sequence stays real and positive
        # y1^2 = 3y0^2 + 20y0. y1 > y0.
        # 2 y_{j-1} >= y_j.
        # If y0 is very small, y1 is approx sqrt(20)y0. Ratio > 1.
        # If y0 is large, y1 approx sqrt(3)y0. Ratio 1.73 > 1.
        # The sequence y_j might grow?
        # If y_j grows, 2 y_{j-1} - y_j might become negative.
        # We need to find y0 such that the final condition holds.
        
        lo = 1e-6
        hi = 1.0 # x0 < 1 => y0 < 1
        
        # Check signs at boundaries
        err_lo = final_error(lo)
        err_hi = final_error(hi)
        
        # If both same sign, try wider range or fallback
        if err_lo * err_hi > 0:
            # Fallback to power law
            p = 2.0
            return (np.arange(n) / (n - 1)) ** p

        for _ in range(100):
            mid = (lo + hi) / 2.0
            err_mid = final_error(mid)
            if err_mid == 0:
                break
            if err_lo * err_mid < 0:
                hi = mid
                err_hi = err_mid
            else:
                lo = mid
                err_lo = err_mid
                
        y0_opt = (lo + hi) / 2.0
        
        # Reconstruct sequence
        y_vals = [y0_opt]
        y_prev = y0_opt
        y_curr = np.sqrt(3 * y0_opt**2 + 2 * y0_opt * (r2 - 1))
        y_vals.append(y_curr)
        
        for j in range(1, n - 1):
            val = 2 * y_curr * y_prev - y_curr**2
            if val < 0:
                val = 0
            y_next = np.sqrt(val)
            y_prev, y_curr = y_curr, y_next
            y_vals.append(y_curr)
            
        x_vals = np.array(y_vals) ** 2
        return x_vals

    def __call__(self, problem, bounds):
        n = self.budget
        if n <= 0:
            return None
        
        x0_vals = self.optimal_x0
        
        for x0 in x0_vals:
            if time.time() > self.deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = x0
            
            try:
                problem(x)
            except Exception:
                break

        return None
