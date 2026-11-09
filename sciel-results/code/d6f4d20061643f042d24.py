import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0 # 10s safety margin

        x_best = np.zeros(self.dim)
        f_best = func(x_best)
        calls = 1
        
        # If budget is very small, we might not even finish the vertex detection
        if self.budget < 12:
             # Fallback to simple random or just return
             # But budget is 500, so this is just for safety
             for _ in range(self.budget - calls):
                 if time.time() > deadline: break
                 x = np.random.uniform(-5, 5, self.dim)
                 fx = func(x)
                 if fx < f_best:
                     f_best = fx
                     x_best = x
                 calls += 1
             return x_best, f_best

        # Determine signs of gradients
        # f(x) is linear. Min is at a vertex.
        # We probe each dimension.
        probe_delta = 0.1
        vertex = np.zeros(self.dim)
        
        for i in range(self.dim):
            if time.time() > deadline:
                break
            
            x_plus = x_best.copy() # Using x_best (0) as base
            x_minus = x_best.copy()
            x_plus[i] += probe_delta
            x_minus[i] -= probe_delta
            
            # Clip to box, though 0 +/- 0.1 is inside [-5, 5]
            f_plus = func(x_plus)
            calls += 1
            f_minus = func(x_minus)
            calls += 1
            
            if f_plus < f_minus:
                # Gradient is negative (c_i < 0), so min is at max x_i
                vertex[i] = 5.0
            else:
                # Gradient is positive (c_i > 0), so min is at min x_i
                vertex[i] = -5.0
            
            # Update best if the probe was better (unlikely for linear, but safe)
            if f_plus < f_best:
                f_best = f_plus
                x_best = x_plus
            if f_minus < f_best:
                f_best = f_minus
                x_best = x_minus

        # Evaluate the candidate vertex
        if calls < self.budget:
            f_vertex = func(vertex)
            calls += 1
            if f_vertex < f_best:
                f_best = f_vertex
                x_best = vertex
            else:
                # If vertex wasn't better, something is wrong (non-linear?)
                # Keep x_best
                pass

        # Use remaining budget for local search around the best point
        # to handle any potential non-linearities or numerical issues
        sigma = 0.5
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            # Simple random local search
            # Or Nelder-Mead? 
            # For linear, the vertex is exact. 
            # If we assume strict linearity, we can just break.
            # But to be robust, let's do a few random steps from x_best.
            # Actually, if it's linear, any other point is worse.
            # So we can just break to save time, or do nothing.
            # The prompt says "repeating the last n_i if fewer calls are made".
            # So breaking early is fine and good for wall-clock time.
            
            # However, if the function is NOT perfectly linear (e.g. f5 is linear, 
            # but maybe there's a trick?), a local search is safer.
            # Let's do a small number of local steps just in case.
            
            y = x_best + sigma * np.random.randn(self.dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < f_best:
                f_best = fy
                x_best = y
                sigma *= 1.5
            else:
                sigma *= 0.8
                if sigma < 1e-5:
                    break
                    
        return x_best, f_best
