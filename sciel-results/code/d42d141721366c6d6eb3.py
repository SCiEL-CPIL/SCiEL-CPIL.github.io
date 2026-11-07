import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def __call__(self, func):
        self.deadline = time.time() + 290
        self.calls = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = np.inf

        def wrapped_func(x):
            if self.calls >= self.budget or time.time() > self.deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            self.calls += 1
            f = float(func(x))
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        def run_nm(x0, local_budget):
            n = self.dim
            # Build simplex with better initial spread
            simplex = [x0.copy()]
            base_delta = 0.5
            for i in range(n):
                x_i = x0.copy()
                if x0[i] + base_delta > 5:
                    x_i[i] -= base_delta
                else:
                    x_i[i] += base_delta
                simplex.append(x_i)
            
            f_vals = [wrapped_func(v) for v in simplex]
            if np.any(np.isinf(f_vals)):
                best_idx = np.argmin(f_vals)
                return simplex[best_idx], f_vals[best_idx]

            iterations = 0
            max_iter = local_budget // (n + 2)
            
            while iterations < max_iter and self.calls < self.budget and time.time() < self.deadline:
                iterations += 1
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]
                
                if f_vals[-1] - f_vals[0] < 1e-10:
                    break
                    
                x_worst = simplex[-1]
                x_best = simplex[0]
                x_centroid = np.mean(simplex[:-1], axis=0)
                
                x_r = x_centroid + (x_centroid - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = wrapped_func(x_r)
                
                if np.isinf(f_r):
                    break
                    
                if f_r < f_vals[0]:
                    x_e = x_centroid + 2 * (x_centroid - x_worst)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = wrapped_func(x_e)
                    if np.isinf(f_e):
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                elif f_r < f_vals[-2]:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                else:
                    x_c = x_centroid + 0.5 * (x_worst - x_centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = wrapped_func(x_c)
                    if np.isinf(f_c):
                        break
                    if f_c < f_r:
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
                            f_vals[i] = wrapped_func(simplex[i])
                            if np.isinf(f_vals[i]):
                                break
                        if np.isinf(f_vals[i]):
                            break
                            
            best_idx = np.argmin(f_vals)
            return simplex[best_idx], f_vals[best_idx]

        def run_adaptive_cd(x0, local_budget):
            x = x0.copy()
            f = wrapped_func(x)
            if np.isinf(f):
                return x, f
            
            # Adaptive step size coordinate descent
            step = 1.0
            min_step = 1e-6
            max_step = 2.0
            no_improve_count = 0
            
            while self.calls < self.budget and time.time() < self.deadline and step > min_step:
                improved_this_pass = False
                for i in range(self.dim):
                    if self.calls >= self.budget or time.time() > self.deadline:
                        break
                    
                    # Try positive direction
                    x_plus = x.copy()
                    x_plus[i] += step
                    x_plus = np.clip(x_plus, -5, 5)
                    f_plus = wrapped_func(x_plus)
                    if not np.isinf(f_plus) and f_plus < f:
                        x = x_plus
                        f = f_plus
                        improved_this_pass = True
                        no_improve_count = 0
                        continue
                    
                    # Try negative direction
                    x_minus = x.copy()
                    x_minus[i] -= step
                    x_minus = np.clip(x_minus, -5, 5)
                    f_minus = wrapped_func(x_minus)
                    if not np.isinf(f_minus) and f_minus < f:
                        x = x_minus
                        f = f_minus
                        improved_this_pass = True
                        no_improve_count = 0
                        continue
                
                if not improved_this_pass:
                    no_improve_count += 1
                    if no_improve_count >= 2:
                        step *= 0.5
                        no_improve_count = 0
                    else:
                        break
            return x, f

        # Multi-start strategy with adaptive budget allocation
        rng = np.random.default_rng(42)
        
        # Phase 1: Broad exploration with NM from multiple starts
        starts = [
            np.zeros(self.dim),
            rng.uniform(-3, 3, self.dim),
            rng.uniform(-4, 4, self.dim),
            rng.uniform(-5, 5, self.dim),
            rng.uniform(-5, 5, self.dim),
        ]
        
        # Allocate budget: ~60% to NM, ~40% to CD refinement
        nm_budget_each = 40
        cd_budget_each = 20
        
        for i, x0 in enumerate(starts):
            if self.calls >= self.budget or time.time() > self.deadline:
                break
            x, f = run_nm(x0, nm_budget_each)
            if self.calls >= self.budget or time.time() > self.deadline:
                break
            x, f = run_adaptive_cd(x, cd_budget_each)
        
        # Phase 2: Final polish on best point with remaining budget
        remaining = self.budget - self.calls
        if remaining > 10 and time.time() < self.deadline:
            x, f = run_adaptive_cd(self.best_x.copy(), remaining)
        
        return self.best_x, self.best_f
