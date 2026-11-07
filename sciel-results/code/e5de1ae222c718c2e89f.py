import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        calls = 0
        deadline = time.time() + 290

        def evaluate(x):
            nonlocal calls
            if time.time() > deadline:
                raise TimeoutError()
            calls += 1
            return float(func(np.asarray(x, dtype=np.float64)))

        # Phase 1: Estimate gradient using large step from origin
        # For linear f(x) = b0 + c^T x, evaluating at x_i = 5*e_i gives
        # f(5*e_i) = b0 + 5*c_i, so sign of (f(5*e_i) - f(0)) = sign of c_i
        x0 = np.zeros(dim)
        f0 = evaluate(x0)
        best_x = x0.copy()
        best_f = f0

        # Use step = 5.0 for maximum robustness in sign determination
        step = 5.0
        signs = np.zeros(dim)
        for i in range(dim):
            xi = np.zeros(dim)
            xi[i] = step
            fi = evaluate(xi)

            if fi < best_f:
                best_f = fi
                best_x = xi.copy()

            if fi < f0:
                signs[i] = 1.0   # gradient component negative, go to +5
            else:
                signs[i] = -1.0  # gradient component positive, go to -5

        # Phase 2: Evaluate the predicted optimal vertex immediately
        # This is the global minimum for a linear function on a box
        x_vertex = np.where(signs == 1.0, 5.0, -5.0)
        f_vertex = evaluate(x_vertex)

        if f_vertex < best_f:
            best_f = f_vertex
            best_x = x_vertex.copy()

        # Phase 3: Use remaining budget for safety verification
        # Check adjacent vertices to confirm our sign estimates are correct
        # For each dimension, flip the sign and check if it's better
        remaining = self.budget - calls
        for _ in range(min(remaining, 10)):
            if time.time() > deadline:
                break
            # Try flipping one dimension at a time from the best vertex
            improved = False
            for i in range(dim):
                x_trial = best_x.copy()
                x_trial[i] = -x_trial[i]  # flip to the other boundary
                f_trial = evaluate(x_trial)
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial.copy()
                    improved = True
                    break  # restart the loop with new best
            if not improved:
                break

        return best_x, best_f
