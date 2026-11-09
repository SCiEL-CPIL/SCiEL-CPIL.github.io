"""Model-and-optimize approach for BBOB f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Model-based optimizer for BBOB-style f5 (Linear slope).
    
    Strategy:
    1. Estimate the linear model parameters (gradient 'a' and offset 'b') 
       using finite differences from the origin.
    2. Generate all 32 vertices of the box [-5, 5]^5.
    3. Sort vertices by the predicted function value from the linear model.
    4. Evaluate vertices in order of predicted value.
    5. This ensures the global minimum (which is at a vertex for linear functions)
       is found very early, maximizing the AOCC score.
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
            
            # Optional: If we found a very good value, we could stop early,
            # but checking all 32 is cheap and guarantees robustness.
            # However, if the first one is the min, we don't need the rest for the score,
            # but we do need to make calls? No, we can stop.
            # But "repeating the last n_i" means stopping early is fine.
            # Let's stop if we are confident?
            # For a linear function, the first one in the sorted list is the min.
            # If we stop after 1 vertex evaluation, we use 7 calls total.
            # If we evaluate all 32, we use 38 calls.
            # AOCC is better if we stop earlier?
            # n_i is 0 for i >= 7.
            # Mean(n_1..n_500) = (sum(n_1..n_6) + 0) / 500.
            # If we stop at 7, n_8..n_500 are n_7=0.
            # So stopping early is better or equal.
            # However, if the gradient estimate was slightly wrong, the true min might be the 2nd or 3rd in the list.
            # So we should evaluate a few to be sure.
            # Let's evaluate all 32 to be safe. It's only 31 extra calls.
            # The cost in AOCC of making 31 extra calls where n_i=0 is zero.
            # So we should evaluate all 32.

        # 6. If budget remains, do nothing. The best is already found.
        # Returning best_x, best_fx.
        
        return best_x, best_fx

# EVOLVE-BLOCK-END
