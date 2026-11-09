import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        x = np.zeros(self.dim)
        fx = func(x)
        calls = 1
        sigma = 2.0
        history = []
        no_improve_count = 0
        best_x = x.copy()
        best_f = fx

        deadline = time.time() + 290

        while calls < self.budget and time.time() < deadline:
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                history.append(True)
                no_improve_count = 0
            else:
                history.append(False)
                no_improve_count += 1

            if len(history) > 15:
                history.pop(0)
            
            if len(history) >= 15:
                rate = np.mean(history)
                sigma *= 1.0 + 0.3 * (rate - 0.2)
                sigma = np.clip(sigma, 1e-4, 5.0)
            
            # Stagnation handling
            if no_improve_count > 50:
                sigma *= 0.5
                no_improve_count = 0

            if fy < best_f:
                best_f = fy
                best_x = y.copy()

        return best_x, best_f
