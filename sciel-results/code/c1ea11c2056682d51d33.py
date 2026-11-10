import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Phase 1: Structured Grid on first 2 variables
        n_grid = 10
        grid_x1 = np.linspace(lo[0], hi[0], n_grid)
        grid_x2 = np.linspace(lo[1], hi[1], n_grid)
        
        X_grid = []
        for x1 in grid_x1:
            for x2 in grid_x2:
                x = np.zeros(self.dim)
                x[0] = x1
                x[1] = x2
                if self.dim > 2:
                    x[2:] = (lo[2:] + hi[2:]) / 2
                X_grid.append(x)
                
        X_grid = np.array(X_grid)
        F_grid = np.array([problem(x) for x in X_grid])
        
        used = len(X_grid)
        
        def get_non_dominated_indices(F):
            nd_indices = []
            n = len(F)
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_indices.append(i)
            return nd_indices

        nd_indices = get_non_dominated_indices(F_grid)
        
        if len(nd_indices) > 10:
            sums = np.sum(F_grid[nd_indices], axis=1)
            best_nd = np.argsort(sums)[:10]
            nd_indices = [nd_indices[i] for i in best_nd]
            
        remaining = self.budget - used
        if remaining > 0 and nd_indices:
            points_to_refine = [X_grid[i] for i in nd_indices]
            evals_per_point = remaining // len(points_to_refine)
            if evals_per_point < 1:
                evals_per_point = 1
                
            all_X = X_grid.copy()
            all_F = F_grid.copy()
            
            for i, base_x in enumerate(points_to_refine):
                base_f = F_grid[nd_indices[i]]
                for _ in range(evals_per_point):
                    if used >= self.budget:
                        break
                    sigma = 0.05 * range_
                    noise = np.random.normal(0, sigma)
                    new_x = np.clip(base_x + noise, lo, hi)
                    new_f = problem(new_x)
                    used += 1
                    
                    if np.sum(new_f) < np.sum(base_f):
                        base_x = new_x
                        base_f = new_f
                        all_X = np.vstack([all_X, new_x])
                        all_F = np.vstack([all_F, new_f])
                        
        nd_indices_final = get_non_dominated_indices(all_F)
        return all_X[nd_indices_final]
