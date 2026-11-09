import numpy as np

class Optimizer:
    """Multi-Start Nelder-Mead optimizer on the box [-5, 5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        total_budget = self.budget
        
        self.global_best_x = None
        self.global_best_f = float('inf')
        self.calls_count = 0

        def track_func(x):
            self.calls_count += 1
            f_val = func(x)
            if f_val < self.global_best_f:
                self.global_best_f = f_val
                self.global_best_x = x.copy()
            return f_val

        def run_nm(x0, alloc):
            # x0: initial point
            # alloc: number of function calls allowed for this instance
            
            if self.calls_count + 1 > total_budget:
                return

            f0 = track_func(x0)
            if self.calls_count >= total_budget:
                return

            simplex = [x0.copy()]
            fvals = [f0]
            
            # Initialize simplex
            for i in range(n):
                if self.calls_count >= total_budget:
                    return
                xi = x0.copy()
                # Adaptive step size based on dimension?
                # Use 0.5 as before
                xi[i] += 0.5
                fi = track_func(xi)
                simplex.append(xi)
                fvals.append(fi)
                
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            while self.calls_count < total_budget:
                # Check if we have enough budget for a full iteration?
                # An iteration might take 1, 2, or n+1 calls.
                # To be safe, just check at each call.
                
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                centroid = np.mean(simplex[:n], axis=0)
                x_worst = simplex[-1]
                f_worst = fvals[-1]
                
                # Reflection
                if self.calls_count >= total_budget:
                    break
                x_r = centroid + alpha * (centroid - x_worst)
                f_r = track_func(x_r)
                
                if f_r < fvals[0]:
                    # Expansion
                    if self.calls_count >= total_budget:
                        break
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = track_func(x_e)
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                        
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if self.calls_count >= total_budget:
                        break
                    if f_r < f_worst:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (x_worst - centroid)
                        
                    f_c = track_func(x_c)
                    
                    if f_c < min(f_r, f_worst):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            if self.calls_count >= total_budget:
                                return
                            x_s = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_s = track_func(x_s)
                            simplex[i] = x_s
                            fvals[i] = f_s

        # Multi-start
        num_starts = 5
        budget_per_start = total_budget // num_starts
        
        starts = [np.zeros(n)]
        for _ in range(num_starts - 1):
            starts.append(np.random.uniform(-5, 5, n))
            
        for x0 in starts:
            if self.calls_count >= total_budget:
                break
            run_nm(x0, budget_per_start)
            
        return self.global_best_x, self.global_best_f
