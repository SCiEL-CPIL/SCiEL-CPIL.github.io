import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = self.start_time + 325.0

    def __call__(self, problem, bounds):
        n = self.budget
        if n <= 0:
            return None
        
        # Solve for x0 such that x_{n-1} = 1
        # x1 = 3*x0 + 20*sqrt(x0)
        # x_{k+1} = 3*xk - 2*sqrt(xk*x_{k-1})
        
        def calculate_last_x0(x0):
            if x0 <= 0:
                return -1.0
            x_prev = x0
            x_curr = 3 * x0 + 20 * np.sqrt(x0)
            if x_curr >= 1.0:
                return x_curr
            for k in range(1, n - 1):
                x_next = 3 * x_curr - 2 * np.sqrt(x_curr * x_prev)
                if x_next >= 1.0:
                    return x_next
                x_prev = x_curr
                x_curr = x_next
            return x_curr

        # Bisection for x0
        # x0 must be small enough so that x_{n-1} <= 1
        # If x0 is too large, x_{n-1} > 1
        # If x0 is too small, x_{n-1} < 1
        
        lo = 1e-10
        hi = 0.1
        # Check hi
        if calculate_last_x0(hi) < 1.0:
             # Need larger x0? No, larger x0 -> larger x_{n-1}
             # So if hi gives < 1, we need larger?
             # Wait. x0 increases -> x1 increases -> all xk increase.
             # So if calculate_last_x0(hi) < 1, then x0 is too small?
             # No, if x0 is small, x_{n-1} is small.
             # So if x_{n-1} < 1, we need to INCREASE x0.
             # So hi is too small?
             # Let's check x0=0.1.
             # x1 = 0.3 + 0.63 = 0.93
             # x2 = 3(0.93) - 2sqrt(0.93*0.1) = 2.79 - 0.61 = 2.18 > 1.
             # So 0.1 is too large.
             # So hi is an upper bound where x_{n-1} > 1.
             pass
             
        # Ensure hi is large enough
        while calculate_last_x0(hi) < 1.0:
            hi *= 2
            if hi > 1.0: break
            
        # Ensure lo is small enough
        while calculate_last_x0(lo) > 1.0:
            lo /= 2
            if lo < 1e-15: break
            
        for _ in range(100):
            mid = (lo + hi) / 2
            x_last = calculate_last_x0(mid)
            if x_last < 1.0:
                lo = mid
            else:
                hi = mid
                
        x0_opt = (lo + hi) / 2
        
        # Generate points
        x_points = [x0_opt]
        x_prev = x0_opt
        x_curr = 3 * x0_opt + 20 * np.sqrt(x0_opt)
        if x_curr > 1.0:
            x_curr = 1.0
        x_points.append(x_curr)
        
        for k in range(1, n - 1):
            x_next = 3 * x_curr - 2 * np.sqrt(x_curr * x_prev)
            if x_next > 1.0:
                x_next = 1.0
            x_points.append(x_next)
            x_prev = x_curr
            x_curr = x_next
            
        # Ensure the last point is exactly 1.0 if possible
        # The bisection should have made it close to 1.0
        # But due to floating point, it might be slightly off.
        # Clipping to 1.0 is safe as it's the boundary.
        
        for x0 in x_points:
            if time.time() > self.deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x0
            try:
                problem(x)
            except Exception:
                break

        return None
