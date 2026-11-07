import numpy as np
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.func = None

    def __call__(self, func):
        self.func = func
        self.calls = 0
        rng = np.random.default_rng(42)
        
        # Initial point
        x = rng.uniform(-5, 5, self.dim)
        fx = self._safe_call(x)
        
        # Directions: standard basis
        directions = [np.eye(self.dim)[i] for i in range(self.dim)]
        
        # Powell's Method
        # We run cycles until budget is exhausted or convergence
        # A cycle consists of d line searches.
        # We need to be careful with the number of calls per line search.
        # minimize_scalar might use many calls. We should limit them or use a custom line search.
        
        # Custom line search to control budget better
        def line_search(x, d, fx):
            # Determine valid alpha range
            # x + alpha * d in [-5, 5]
            # alpha * d_i in [-5 - x_i, 5 - x_i]
            # If d_i > 0: alpha in [(-5 - x_i)/d_i, (5 - x_i)/d_i]
            # If d_i < 0: alpha in [(5 - x_i)/d_i, (-5 - x_i)/d_i]
            # If d_i == 0: no constraint
            
            alpha_min = -np.inf
            alpha_max = np.inf
            
            for i in range(self.dim):
                if abs(d[i]) < 1e-12:
                    continue
                if d[i] > 0:
                    a_min = (-5 - x[i]) / d[i]
                    a_max = (5 - x[i]) / d[i]
                else:
                    a_min = (5 - x[i]) / d[i]
                    a_max = (-5 - x[i]) / d[i]
                
                alpha_min = max(alpha_min, a_min)
                alpha_max = min(alpha_max, a_max)
            
            if alpha_min >= alpha_max:
                # No movement possible in this direction?
                return x, fx, 0.0

            # Use scipy.optimize.minimize_scalar with bounded method
            # This uses Brent's method, which is robust.
            # We need to limit the number of function calls.
            # However, minimize_scalar doesn't have a direct 'maxiter' that stops early 
            # based on an external counter easily without wrapping.
            
            # Let's wrap the function to count calls and raise an exception if budget exceeded?
            # Or just check after.
            
            def objective(alpha):
                if self.calls >= self.budget:
                    return np.inf
                x_new = x + alpha * d
                return self._safe_call(x_new)
            
            # We can't easily pass a stop condition to minimize_scalar.
            # Instead, we will implement a simple ternary search or use minimize_scalar 
            # and hope it's efficient. For a quadratic, it's very efficient (converges in ~5-10 iterations).
            
            res = scipy.optimize.minimize_scalar(objective, bounds=(alpha_min, alpha_max), method='bounded', options={'maxiter': 20})
            
            alpha_opt = res.x
            x_new = x + alpha_opt * d
            fx_new = self._safe_call(x_new)
            
            return x_new, fx_new, fx - fx_new

        # Main loop
        while self.calls < self.budget:
            x_old = x.copy()
            fx_old = fx
            
            improvement = 0.0
            worst_dir_idx = -1
            worst_improvement = np.inf
            
            for i, d in enumerate(directions):
                if self.calls >= self.budget:
                    break
                
                x, fx, imp = line_search(x, d, fx)
                improvement += imp
                
                if imp < worst_improvement:
                    worst_improvement = imp
                    worst_dir_idx = i
            
            if self.calls >= self.budget:
                break
            
            # Check convergence
            if abs(fx_old - fx) < 1e-12:
                break
            
            # Update directions
            # New direction is x - x_old
            new_dir = x - x_old
            norm_new_dir = np.linalg.norm(new_dir)
            if norm_new_dir < 1e-12:
                continue
                
            # Line search along new_dir
            x_new_dir, fx_new_dir, imp_new = line_search(x_old, new_dir, fx_old)
            
            if self.calls >= self.budget:
                break
            
            # If the improvement along the new direction is better than the worst in the cycle,
            # replace the worst direction with the new one
            if imp_new > worst_improvement:
                # Normalize the new direction? Powell's method usually doesn't strictly require normalization
                # but it helps with step size consistency.
                directions[worst_dir_idx] = new_dir
                x, fx = x_new_dir, fx_new_dir
            else:
                # Keep x, fx from the end of the cycle
                # x, fx are already updated from the last line search in the loop
                pass

        return x, fx

    def _safe_call(self, x):
        if self.calls >= self.budget:
            # Return a large value to discourage further search?
            # But we shouldn't call if we are out.
            # The caller should check.
            pass
        self.calls += 1
        return self.func(x)
