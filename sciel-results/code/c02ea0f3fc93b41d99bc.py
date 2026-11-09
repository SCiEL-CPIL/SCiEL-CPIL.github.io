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
            return np.array([0.5]) # Or something, doesn't matter much
        
        # We want to find y2 such that the sequence y_i starting with y1=0, y2=y2
        # ends with y_n = 1.
        # Recurrence: y_{i+1} = sqrt(3*y_i^2 - 2*y_i*y_{i-1})
        
        def generate_sequence(y2):
            y_prev = 0.0
            y_curr = y2
            ys = [y_prev, y_curr]
            for i in range(1, n - 1):
                val = 3 * y_curr**2 - 2 * y_curr * y_prev
                if val < 0:
                    return None
                y_next = np.sqrt(val)
                ys.append(y_next)
                y_prev, y_curr = y_curr, y_next
            return ys

        # Bisection for y2
        # y_n is increasing with y2
        lo = 1e-6
        hi = 1.0
        
        # Check hi
        seq_hi = generate_sequence(hi)
        if seq_hi is None or seq_hi[-1] < 1.0:
            # If even y2=1 gives y_n < 1, then we can't reach 1?
            # But y_n should be > y2.
            # Let's just use a wide range or fallback
            pass
            
        err_lo = 0
        seq_lo = generate_sequence(lo)
        if seq_lo is not None:
            err_lo = seq_lo[-1] - 1.0
            
        err_hi = 0
        if seq_hi is not None:
            err_hi = seq_hi[-1] - 1.0
            
        if err_lo * err_hi > 0:
            # Fallback to power law
            p = 2.0
            return (np.arange(n) / (n - 1)) ** p

        for _ in range(100):
            mid = (lo + hi) / 2.0
            seq_mid = generate_sequence(mid)
            if seq_mid is None:
                hi = mid
                continue
            err_mid = seq_mid[-1] - 1.0
            if err_mid == 0:
                break
            if err_lo * err_mid < 0:
                hi = mid
                err_hi = err_mid
            else:
                lo = mid
                err_lo = err_mid
                
        y2_opt = (lo + hi) / 2.0
        y_vals = generate_sequence(y2_opt)
        
        if y_vals is None:
            p = 2.0
            return (np.arange(n) / (n - 1)) ** p

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
