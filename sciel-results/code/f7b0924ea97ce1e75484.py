import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        
    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 250 # 300s limit, leave 50s margin
        
        # Wrapper to count calls and track best
        def eval_func(x):
            if self.calls >= self.budget:
                raise StopIteration
            self.calls += 1
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            return fx
            
        # 1. Initial Exploration
        # Try a few deterministic points to find a good basin
        init_points = [
            np.zeros(self.dim),
            np.ones(self.dim) * 2,
            np.ones(self.dim) * -2,
            np.ones(self.dim) * 1,
            np.ones(self.dim) * -1
        ]
        
        best_init_x = None
        best_init_f = float('inf')
        
        for p in init_points:
            if self.calls >= self.budget or time.time() > deadline:
                break
            try:
                fp = eval_func(p)
                if fp < best_init_f:
                    best_init_f = fp
                    best_init_x = p
            except StopIteration:
                break
                
        if best_init_x is None:
            # If we couldn't even do one call, return zero
            return np.zeros(self.dim), float('inf')

        # 2. Nelder-Mead Optimization
        # Initialize simplex around best_init_x
        # Standard Nelder-Mead uses a simplex of D+1 points.
        # We already have best_init_x. We need D more points.
        # Let's use a small perturbation.
        step = 0.5 # Initial step size for simplex
        simplex = [best_init_x]
        for i in range(self.dim):
            v = best_init_x.copy()
            v[i] += step
            simplex.append(v)
            
        # Evaluate the rest of the simplex
        simplex_f = [best_init_f]
        for v in simplex[1:]:
            if self.calls >= self.budget or time.time() > deadline:
                break
            try:
                fv = eval_func(v)
                simplex_f.append(fv)
            except StopIteration:
                break
        
        # If we didn't finish initializing simplex, just return best
        if len(simplex_f) < self.dim + 1:
             return self.best_x, self.best_f

        # Main Loop
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        while self.calls < self.budget and time.time() < deadline:
            # Sort simplex by function value
            indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in indices]
            simplex_f = [simplex_f[i] for i in indices]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:self.dim], axis=0)
            
            # Reflection
            worst = simplex[-1]
            worst_f = simplex_f[-1]
            
            reflection = centroid + alpha * (centroid - worst)
            try:
                fr = eval_func(reflection)
            except StopIteration:
                break
                
            if fr < simplex_f[0]:
                # Expansion
                expansion = centroid + gamma * (reflection - centroid)
                try:
                    fe = eval_func(expansion)
                except StopIteration:
                    break
                if fe < fr:
                    simplex[-1] = expansion
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = reflection
                    simplex_f[-1] = fr
            elif fr < simplex_f[-2]:
                # Accept Reflection
                simplex[-1] = reflection
                simplex_f[-1] = fr
            else:
                # Contraction
                if fr < worst_f:
                    # Inside contraction
                    contraction = centroid + rho * (worst - centroid)
                else:
                    # Outside contraction
                    contraction = centroid - rho * (centroid - worst)
                    
                try:
                    fc = eval_func(contraction)
                except StopIteration:
                    break
                    
                if fc < min(fr, worst_f):
                    simplex[-1] = contraction
                    simplex_f[-1] = fc
                else:
                    # Shrink
                    for i in range(1, self.dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        try:
                            simplex_f[i] = eval_func(simplex[i])
                        except StopIteration:
                            return self.best_x, self.best_f

        return self.best_x, self.best_f
