import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = None
        best_f = float('inf')
        calls = 0
        
        # Multi-start strategy
        # First restart gets more budget to ensure early progress for AOCC
        restarts = 5
        budgets = [200, 100, 100, 50, 50] # Total 500
        
        for b in budgets:
            if calls >= self.budget:
                break
            
            # Random start
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            calls += 1
            
            if fx < best_f:
                best_x = x.copy()
                best_f = fx
            
            # Local search
            x, fx, calls = self._coordinate_descent(func, x, fx, b - 1, calls, rng)
            
            if fx < best_f:
                best_x = x.copy()
                best_f = fx
                
        return best_x

    def _coordinate_descent(self, func, x, fx, budget, calls, rng):
        steps = np.ones(self.dim) * 0.5
        last_improve = calls
        
        while budget > 0:
            improved = False
            for i in range(self.dim):
                if budget < 2:
                    break
                
                # Try + step
                x_plus = x.copy()
                x_plus[i] = np.clip(x_plus[i] + steps[i], -5, 5)
                f_plus = func(x_plus)
                calls += 1
                budget -= 1
                
                # Try - step
                x_minus = x.copy()
                x_minus[i] = np.clip(x_minus[i] - steps[i], -5, 5)
                f_minus = func(x_minus)
                calls += 1
                budget -= 1
                
                if f_plus < fx:
                    x = x_plus
                    fx = f_plus
                    steps[i] *= 1.5
                    improved = True
                    last_improve = calls
                elif f_minus < fx:
                    x = x_minus
                    fx = f_minus
                    steps[i] *= 1.5
                    improved = True
                    last_improve = calls
                else:
                    steps[i] *= 0.5
            
            if not improved:
                # If no improvement in a full cycle, check stagnation
                if calls - last_improve > 10:
                    break
            else:
                # If improved, reset stagnation counter logic? 
                # No, last_improve is updated.
                pass
                
        return x, fx, calls
