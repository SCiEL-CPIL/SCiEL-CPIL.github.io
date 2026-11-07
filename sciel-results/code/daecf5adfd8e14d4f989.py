import numpy as np

class Optimizer:
    """Nelder-Mead optimizer with multi-start initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        total_budget = self.budget
        
        eval_count = 0
        best_x = None
        best_f = float('inf')
        
        # Phase 1: Multi-start initialization
        # Use 10% of budget for initialization, at least 5 points, at most 20
        init_budget = max(5, min(20, total_budget // 10))
        
        # Generate initial candidates
        # Mix of random and centered
        candidates = []
        # Center
        candidates.append(np.zeros(n))
        # Random
        for _ in range(init_budget - 1):
            candidates.append(rng.uniform(-5, 5, size=n))
            
        # Evaluate candidates
        for x in candidates:
            if eval_count >= total_budget:
                break
            x = np.clip(x, -5, 5)
            f = func(x)
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
        if eval_count >= total_budget:
            return best_x, best_f
            
        # Phase 2: Nelder-Mead from best_x
        # Initialize simplex
        # Center at best_x
        # Initial step size
        step = 0.5
        
        simplex = np.zeros((n + 1, n))
        simplex[0] = best_x
        for i in range(n):
            simplex[i + 1] = best_x
            simplex[i + 1][i] += step
            
        # Evaluate simplex
        simplex_f = np.zeros(n + 1)
        for i in range(n + 1):
            if eval_count >= total_budget:
                break
            simplex_f[i] = func(simplex[i])
            eval_count += 1
            if simplex_f[i] < best_f:
                best_f = simplex_f[i]
                best_x = simplex[i].copy()
                
        # NM Parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Main NM Loop
        while eval_count < total_budget:
            # Sort simplex by fitness
            idx = np.argsort(simplex_f)
            simplex = simplex[idx]
            simplex_f = simplex_f[idx]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            
            if eval_count >= total_budget:
                break
            f_r = func(x_r)
            eval_count += 1
            
            if f_r < best_f:
                best_f = f_r
                best_x = x_r.copy()
                
            if simplex_f[0] <= f_r < simplex_f[-2]:
                # Replace worst
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            elif f_r < simplex_f[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                
                if eval_count >= total_budget:
                    break
                f_e = func(x_e)
                eval_count += 1
                
                if f_e < best_f:
                    best_f = f_e
                    best_x = x_e.copy()
                    
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            else:
                # Contraction
                # Outside contraction if f_r < simplex_f[-1], else inside
                if f_r < simplex_f[-1]:
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    x_c = centroid - rho * (simplex[-1] - centroid)
                    
                x_c = np.clip(x_c, -5, 5)
                
                if eval_count >= total_budget:
                    break
                f_c = func(x_c)
                eval_count += 1
                
                if f_c < best_f:
                    best_f = f_c
                    best_x = x_c.copy()
                    
                if f_c < min(f_r, simplex_f[-1]):
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        
                        if eval_count >= total_budget:
                            break
                        simplex_f[i] = func(simplex[i])
                        eval_count += 1
                        if simplex_f[i] < best_f:
                            best_f = simplex_f[i]
                            best_x = simplex[i].copy()
                            
        return best_x, best_f
