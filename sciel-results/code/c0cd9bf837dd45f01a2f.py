import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
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

        # 1. Initialization: Sobol + Random
        n_sobol = 30
        n_random = 10
        n_init = n_sobol + n_random
        
        # Generate Sobol points
        sobol_engine = qmc.Sobol(d=dim, scramble=True)
        sobol_points = sobol_engine.random(n_sobol)
        # Scale to bounds
        sobol_points = lo + (hi - lo) * sobol_points
        
        for i in range(n_sobol):
            if time.time() > deadline: break
            eval_point(sobol_points[i])
            
        for i in range(n_random):
            if time.time() > deadline: break
            x = lo + (hi - lo) * np.random.rand(dim)
            eval_point(x)
            
        if not F:
            return X

        F_arr = np.array(F)
        
        # 2. Local Search from Diverse Non-Dominated Points
        nd_idx = get_non_dominated_indices(F_arr)
        
        # Select diverse starting points for local search
        # We want points that are good in different objectives
        # Simple heuristic: pick points that are closest to the ideal point in different normalized directions
        # Or just pick the top K non-dominated points based on sum of objectives
        
        if len(nd_idx) > 0:
            nd_F = F_arr[nd_idx]
            # Normalize objectives to [0, 1] for selection
            mins = nd_F.min(axis=0)
            maxs = nd_F.max(axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1
            norm_F = (nd_F - mins) / ranges
            
            # Pick points that minimize the sum of normalized objectives
            # This tends to pick points in the "center" of the front, which are often good starting points
            # To get diversity, we can pick the best one for each objective
            start_points_idx = []
            for obj_i in range(n_obj):
                # Find index in nd_idx that minimizes this specific objective
                best_local_idx = np.argmin(nd_F[:, obj_i])
                global_idx = nd_idx[best_local_idx]
                if global_idx not in start_points_idx:
                    start_points_idx.append(global_idx)
            
            # If we have fewer than 3 start points, add the best overall
            if len(start_points_idx) < 3:
                best_sum_idx = np.argmin(norm_F.sum(axis=1))
                global_idx = nd_idx[best_sum_idx]
                if global_idx not in start_points_idx:
                    start_points_idx.append(global_idx)
            
            # Limit to 5 start points max to save budget
            start_points_idx = start_points_idx[:5]
            
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
                        # L-BFGS-B is efficient for this size
                        # maxfun=10 ensures we don't spend too much on one point
                        minimize(objective, x0, method='L-BFGS-B',
                                 bounds=list(zip(lo, hi)),
                                 options={'maxiter': 10, 'maxfun': 10})
                    except Exception:
                        pass

        # 3. Final Polish / Evolutionary Step with Remaining Budget
        # Re-evaluate non-dominated set
        F_arr = np.array(F)
        nd_idx = get_non_dominated_indices(F_arr)
        
        if used < self.budget and time.time() < deadline:
            # Use remaining budget for a simple mutation-based search around the best points
            # Take the top 10 non-dominated points
            if len(nd_idx) > 0:
                nd_F = F_arr[nd_idx]
                mins = nd_F.min(axis=0)
                maxs = nd_F.max(axis=0)
                ranges = maxs - mins
                ranges[ranges == 0] = 1
                norm_F = (nd_F - mins) / ranges
                best_sum_idx = np.argsort(norm_F.sum(axis=1))[:10]
                elite_idx = [nd_idx[i] for i in best_sum_idx]
            else:
                elite_idx = list(range(min(10, len(F))))
            
            # Simple loop: mutate elite points
            while used < self.budget and time.time() < deadline:
                # Pick a random elite
                parent_idx = elite_idx[np.random.randint(len(elite_idx))]
                x_parent = X[parent_idx]
                
                # Gaussian mutation
                sigma = 0.1 * (hi - lo)
                x_child = x_parent + sigma * np.random.randn(dim)
                x_child = np.clip(x_child, lo, hi)
                
                f_child = eval_point(x_child)
                
                if f_child is not None:
                    # If the child is non-dominated, add it to elite list (optional, keeps it fresh)
                    # For simplicity, we just keep evaluating
                    pass
                else:
                    break

        return X
