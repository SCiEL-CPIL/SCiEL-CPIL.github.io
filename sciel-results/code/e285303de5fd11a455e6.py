import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self._call_count = 0
        self._best_f = np.inf
        self._best_x = None

        def count_func(x):
            self._call_count += 1
            # Clip to box
            x = np.clip(x, -5, 5)
            fx = func(x)
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = x.copy()
            return fx

        # Strategy: Multi-start Nelder-Mead
        # We have 500 calls.
        # Nelder-Mead in 5D needs ~ (5+1) * iterations.
        # Let's do 2 restarts.
        # Each restart gets ~250 calls.
        
        # Initial simplex generation
        def make_simplex(center, alpha=0.5, beta=0.5):
            # Center is dim-vector
            # Points: center, center + alpha*e_i, center - beta*e_i? 
            # Standard: center, center + delta*e_i
            delta = 0.1
            simplex = [center.copy()]
            for i in range(self.dim):
                p = center.copy()
                p[i] += delta
                simplex.append(p)
            return simplex

        # Nelder-Mead implementation
        def nelder_mead(func_eval, x0, budget_left):
            # x0 is the center
            # Initialize simplex
            delta = 1.0
            simplex = [x0.copy()]
            for i in range(self.dim):
                p = x0.copy()
                p[i] += delta
                simplex.append(p)
            
            # Evaluate initial simplex
            fvals = []
            for p in simplex:
                if self._call_count >= budget_left:
                    break
                fvals.append(func_eval(p))
            
            if len(fvals) < self.dim + 1:
                return simplex[0], fvals[0] if fvals else np.inf

            # Sort simplex by fitness
            def sort_simplex():
                indices = np.argsort(fvals)
                simplex = [simplex[i] for i in indices]
                fvals = [fvals[i] for i in indices]
                return simplex, fvals
            
            simplex, fvals = sort_simplex()
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            while self._call_count < budget_left:
                # Reflection
                x_worst = simplex[-1]
                f_worst = fvals[-1]
                
                # Centroid of all but worst
                x_c = np.mean(simplex[:-1], axis=0)
                x_r = x_c + alpha * (x_c - x_worst)
                x_r = np.clip(x_r, -5, 5)
                
                if self._call_count >= budget_left: break
                f_r = func_eval(x_r)
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = x_c + gamma * (x_r - x_c)
                    x_e = np.clip(x_e, -5, 5)
                    if self._call_count >= budget_left: break
                    f_e = func_eval(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c_new = x_c + rho * (x_r - x_c)
                    else:
                        # Inside contraction
                        x_c_new = x_c - rho * (x_worst - x_c)
                    
                    x_c_new = np.clip(x_c_new, -5, 5)
                    if self._call_count >= budget_left: break
                    f_c = func_eval(x_c_new)
                    
                    if f_c < min(f_r, f_worst):
                        simplex[-1] = x_c_new
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if self._call_count >= budget_left: break
                            fvals[i] = func_eval(simplex[i])
                        
                        if self._call_count >= budget_left: break

                simplex, fvals = sort_simplex()
                
                # Check convergence
                if np.std(fvals) < 1e-10:
                    break

            return simplex[0], fvals[0]

        # Multi-start
        # Start 1: Center 0
        # Start 2: Random
        rng = np.random.default_rng(42)
        
        # Split budget
        # 250 for first, 250 for second?
        # Or just one strong start?
        # Linear slope might be flat in some directions.
        # Let's try 2 starts.
        
        budget_per_start = self.budget // 2
        
        # Start 1
        x0_1 = np.zeros(self.dim)
        x1, f1 = nelder_mead(count_func, x0_1, self.budget)
        
        # If we still have budget, do another start?
        # The nelder_mead function stops when budget_left is reached or converged.
        # If it converged early, we have budget left.
        
        if self._call_count < self.budget:
            # Start 2: Random point
            x0_2 = rng.uniform(-3, 3, self.dim)
            x2, f2 = nelder_mead(count_func, x0_2, self.budget)

        return self._best_x, self._best_f
