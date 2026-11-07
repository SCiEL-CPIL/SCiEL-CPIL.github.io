import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        X = []
        F = []
        used = 0
        
        def eval_point(x):
            nonlocal used
            if used >= self.budget or time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            X.append(x)
            F.append(f)
            used += 1
            return f

        # 1. Random Initialization
        n_random = 40
        for i in range(n_random):
            if time.time() > deadline: break
            x = lo + (hi - lo) * np.random.rand(dim)
            eval_point(x)
            
        if not F:
            return X

        F_arr = np.array(F)
        
        # Helper to find non-dominated indices
        def get_non_dominated_indices(F_list):
            n = len(F_list)
            if n == 0: return []
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]: continue
                for j in range(n):
                    if i == j: continue
                    if np.all(F_list[j] <= F_list[i]) and np.any(F_list[j] < F_list[i]):
                        dominated[i] = True
                        break
                    if np.all(F_list[i] <= F_list[j]) and np.any(F_list[i] < F_list[j]):
                        dominated[j] = True
            return [i for i in range(n) if not dominated[i]]

        # 2. Gradient Based Local Search
        # Select diverse starting points
        nd_idx = get_non_dominated_indices(F_arr)
        if len(nd_idx) < 4:
             # If not enough non-dominated, pick lowest sum
             sums = F_arr.sum(axis=1)
             nd_idx = np.argsort(sums)[:4]
        
        # Take up to 4 diverse points
        start_points_idx = nd_idx[:4]
        
        for idx in start_points_idx:
            if time.time() > deadline or used >= self.budget:
                break
            x0 = X[idx]
            for obj_i in range(n_obj):
                if time.time() > deadline or used >= self.budget:
                    break
                
                def objective(x):
                    nonlocal used
                    if used >= self.budget or time.time() > deadline:
                        return 0.0
                    x = np.clip(x, lo, hi)
                    f = np.array(problem(x))
                    X.append(x)
                    F.append(f)
                    used += 1
                    return f[obj_i]
                
                try:
                    minimize(objective, x0, method='L-BFGS-B',
                             bounds=list(zip(lo, hi)),
                             options={'maxiter': 20, 'maxfun': 10})
                except Exception:
                    pass

        # 3. Evolutionary Search
        if used < self.budget and time.time() < deadline:
            # Re-evaluate non-dominated set
            F_arr = np.array(F)
            nd_idx = get_non_dominated_indices(F_arr)
            
            # Initialize population with non-dominated points
            pop_size = 10
            if len(nd_idx) < pop_size:
                # Fill with random
                while len(nd_idx) < pop_size and used < self.budget:
                    x = lo + (hi - lo) * np.random.rand(dim)
                    if eval_point(x) is not None:
                        nd_idx.append(used - 1)
            
            pop_idx = nd_idx[:pop_size]
            
            while used < self.budget and time.time() < deadline:
                F_arr = np.array(F)
                nd_idx = get_non_dominated_indices(F_arr)
                if len(nd_idx) < 2:
                    nd_idx = list(range(len(F_arr)))
                
                p1 = nd_idx[np.random.randint(len(nd_idx))]
                p2 = nd_idx[np.random.randint(len(nd_idx))]
                
                x1 = X[p1]
                x2 = X[p2]
                
                # Crossover
                alpha = np.random.rand(dim)
                child = alpha * x1 + (1 - alpha) * x2
                child = np.clip(child, lo, hi)
                
                # Mutation
                if np.random.rand() < 0.3:
                    child += 0.1 * (hi - lo) * np.random.randn(dim)
                    child = np.clip(child, lo, hi)
                
                if eval_point(child) is None:
                    break

        return X
