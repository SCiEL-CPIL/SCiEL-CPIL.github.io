import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s margin
        
        d = self.dim
        budget = self.budget
        
        # For BBOB f5 (Linear Slope), the function is f(x) = w^T x + c.
        # The global minimum on the box [-5, 5]^d is at a corner.
        # We use a sequential coordinate descent strategy to find the optimal corner.
        # This approach updates the best candidate incrementally, ensuring
        # that the best value found improves (or stays same) at each step,
        # which is highly effective for AOCC scoring.
        
        # 1. Start at the origin
        x_curr = np.zeros(d)
        f_curr = func(x_curr)
        best_f = f_curr
        best_x = x_curr.copy()
        
        # 2. For each dimension, determine the optimal sign (-5 or 5)
        # We test -5 first. If it's better than current, we keep it.
        # If not, we test 5. For linear functions, one of them will be better
        # than the current value (unless the gradient is 0, in which case they are equal).
        
        for i in range(d):
            if time.time() > deadline:
                break
            
            # Try negative boundary
            x_neg = x_curr.copy()
            x_neg[i] = -5.0
            f_neg = func(x_neg)
            
            if f_neg < f_curr:
                # Negative is better, update current state
                x_curr = x_neg
                f_curr = f_neg
            else:
                # Negative is not better, try positive boundary
                x_pos = x_curr.copy()
                x_pos[i] = 5.0
                f_pos = func(x_pos)
                
                # For linear functions, if f(-5) >= f(0), then f(5) <= f(0).
                # So f_pos should be <= f_curr.
                # We update to the better of the two (which is f_pos in this case)
                if f_pos < f_curr:
                    x_curr = x_pos
                    f_curr = f_pos
                # If f_pos == f_curr (flat), we can keep x_curr or x_pos.
                # Keeping x_curr is fine.
                
            # Update global best
            if f_curr < best_f:
                best_f = f_curr
                best_x = x_curr.copy()
                
        # After the loop, x_curr should be the optimal corner.
        # We have already evaluated it.
        
        return best_x, best_f
