"""Model-and-optimize approach for BBOB f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Model-based optimizer for BBOB-style f5 (Linear slope).
    
    Strategy:
    1. Start at the origin to estimate the gradient cleanly.
    2. Estimate the gradient using forward differences (dim calls).
    3. Generate all 32 vertices of the box [-5, 5]^5.
    4. Sort vertices by the predicted function value from the linear model.
    5. Evaluate vertices in order of predicted value to find the global minimum
       very early (maximizing AOCC).
    6. Use remaining budget for a simple local search as a safeguard.
    7. Deadline guard to ensure we finish within the limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 15s safety margin
        calls_made = 0
        dim = self.dim
        budget = self.budget

        # 1. Start at origin to estimate gradient cleanly
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls_made += 1
        best_x = x0.copy()
        best_fx = f0

        # 2. Estimate gradient using forward differences
        # f(x) = a^T x + b. f(0) = b.
        # f(h e_i) = a_i h + b.
        # a_i = (f(h e_i) - f(0)) / h
        h = 1.0
        a_est = np.zeros(dim)
        
        for i in range(dim):
            if time.time() > deadline or calls_made >= budget:
                break
            
            xi = x0.copy()
            xi[i] += h
            fi = func(xi)
            calls_made += 1
            
            a_est[i] = (fi - f0) / h
            
            if fi < best_fx:
                best_fx = fi
                best_x = xi.copy()

        # 3. Generate all vertices
        vertices = []
        for mask in range(2 ** dim):
            v = np.array([5.0 if (mask >> i) & 1 else -5.0 for i in range(dim)])
            vertices.append(v)

        # 4. Predict function values at vertices and sort
        # Predicted f(v) = a_est^T v + f0
        predicted_values = []
        for v in vertices:
            pred_val = np.dot(a_est, v) + f0
            predicted_values.append((pred_val, v))
        
        # Sort by predicted value (ascending) to find the minimum first
        predicted_values.sort(key=lambda x: x[0])

        # 5. Evaluate vertices in sorted order
        for pred_val, v in predicted_values:
            if time.time() > deadline or calls_made >= budget:
                break
            
            fv = func(v)
            calls_made += 1
            
            if fv < best_fx:
                best_fx = fv
                best_x = v.copy()

        # 6. If budget remains, do a local search around the best found vertex.
        # This is a safeguard in case the function is not perfectly linear or
        # if there are numerical issues, but primarily it consumes remaining budget
        # without hurting the score (since we've likely already found the min).
        # For a linear function, the minimum is at a vertex, so this should be a no-op
        # in terms of finding a better value, but it ensures we use the time safely.
        
        if calls_made < budget and time.time() < deadline:
            sigma = 0.5
            x_cur = best_x.copy()
            f_cur = best_fx
            
            # We will use the remaining budget for local search steps
            # This is a simple (1+1)-ES style search
            while calls_made < budget and time.time() < deadline:
                y = np.clip(x_cur + sigma * np.random.randn(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < f_cur:
                    x_cur = y
                    f_cur = fy
                    sigma *= 1.1
                    if fy < best_fx:
                        best_fx = fy
                        best_x = y.copy()
                else:
                    sigma *= 0.9
                    if sigma < 1e-4:
                        break

        return best_x, best_fx

# EVOLVE-BLOCK-END
