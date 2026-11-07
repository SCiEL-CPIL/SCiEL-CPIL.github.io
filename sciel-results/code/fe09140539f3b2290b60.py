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
                
                x, fx, imp = self.line_search_scipy(x, d, fx)
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
            new_dir = x - x_old
            norm_new_dir = np.linalg.norm(new_dir)
            if norm_new_dir < 1e-12:
                continue
                
            # Line search along new_dir
            x_new_dir, fx_new_dir, imp_new = self.line_search_scipy(x_old, new_dir, fx_old)
            
            if self.calls >= self.budget:
                break
            
            if imp_new > worst_improvement:
                directions[worst_dir_idx] = new_dir
                x, fx = x_new_dir, fx_new_dir

        return x, fx

    def line_search_scipy(self, x, d, fx):
        # Define a 1D function
        best_x = x
        best_f = fx
        
        def objective(alpha):
            nonlocal best_x, best_f
            if self.calls >= self.budget:
                raise Exception("Budget Exceeded")
            x_candidate = x + alpha * d
            # Clip to box? The task says func clips.
            # But we should avoid going out of bounds if possible to save calls?
            # The task says func clips. So it's safe.
            f_val = self._safe_call(x_candidate)
            if f_val < best_f:
                best_f = f_val
                best_x = x_candidate.copy()
            return f_val

        # Determine bounds for alpha
        # We want x + alpha * d to be in [-5, 5]
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
            return x, fx, 0.0
            
        try:
            res = scipy.optimize.minimize_scalar(
                objective, 
                bounds=(alpha_min, alpha_max), 
                method='brent',
                options={'xatol': 1e-8}
            )
            # If it finished normally, use the result
            # But we already tracked best_x, best_f
            return best_x, best_f, fx - best_f
        except Exception:
            # Budget exceeded or other error
            return best_x, best_f, fx - best_f

    def _safe_call(self, x):
        self.calls += 1
        return self.func(x)
