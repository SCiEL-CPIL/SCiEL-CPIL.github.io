import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # State
        X = []
        F = []
        eval_count = 0
        
        def eval_point(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            fx = np.array(problem(x))
            X.append(x)
            F.append(fx)
            eval_count += 1
            return fx
            
        def check_time():
            return time.time() < deadline - 5.0

        # Helper: Check if a point is non-dominated by the current set F
        def is_non_dominated(f_new, F_current):
            for f_old in F_current:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    return False
            return True

        # Helper: Local Search
        def local_search(x, f, max_steps=10):
            nonlocal eval_count
            current_x = x.copy()
            current_f = f
            # Adaptive step size: start large, shrink if stuck
            step_size = 0.2 * (hi - lo)
            stagnant = 0
            
            for _ in range(max_steps):
                if not check_time() or eval_count >= budget - 10:
                    break
                
                # Generate a neighbor
                neighbor = current_x + np.random.randn(dim) * step_size
                neighbor = np.clip(neighbor, lo, hi)
                
                f_neighbor = eval_point(neighbor)
                if f_neighbor is None:
                    break
                
                if is_non_dominated(f_neighbor, [current_f]):
                    current_x = neighbor
                    current_f = f_neighbor
                    stagnant = 0
                    # Slightly increase step size on success to explore further? Or keep same?
                    # Let's keep same or slightly decrease to refine
                else:
                    stagnant += 1
                    if stagnant > 2:
                        step_size *= 0.7
                        stagnant = 0
            return current_x, current_f

        # 1. Structure-Aware Initialization
        # WFG4 Pareto front is controlled by x[0]. Other vars should be near 0.
        n_init = 30
        X_init = np.zeros((n_init, dim))
        
        # Sample x[0] uniformly
        X_init[:, 0] = np.linspace(lo[0], hi[0], n_init)
        
        # Set other variables to 0 (clipped to bounds)
        for j in range(1, dim):
            # If 0 is within bounds, use 0. Else use closest bound.
            if lo[j] <= 0 <= hi[j]:
                val = 0.0
            else:
                val = lo[j] if abs(lo[j]) < abs(hi[j]) else hi[j]
            X_init[:, j] = val
            
        # Add some random noise to break symmetry and handle cases where 0 is not optimal
        # But keep it small
        X_init += np.random.randn(n_init, dim) * 0.05 * (hi - lo)
        X_init = np.clip(X_init, lo, hi)
        
        # Evaluate and refine
        for i in range(n_init):
            if not check_time():
                break
            fx = eval_point(X_init[i])
            if fx is None:
                break
            X_init[i], fx = local_search(X_init[i], fx, max_steps=5)
            
        # 2. Expansion Phase
        if eval_count < budget:
            # Get non-dominated points
            nd_indices = []
            for i in range(len(F)):
                if is_non_dominated(F[i], F[:i] + F[i+1:]):
                    nd_indices.append(i)
                    
            if len(nd_indices) > 0:
                # Generate offspring from non-dominated points
                # Focus on perturbing the "deceptive" variables more aggressively?
                # Or just general perturbation.
                offspring_to_eval = []
                for idx in nd_indices:
                    parent_x = X[idx]
                    for _ in range(3):
                        if eval_count >= budget - 5:
                            break
                        child_x = parent_x + np.random.randn(dim) * 0.1 * (hi - lo)
                        child_x = np.clip(child_x, lo, hi)
                        offspring_to_eval.append(child_x)
                        
                for ox in offspring_to_eval:
                    if not check_time() or eval_count >= budget - 2:
                        break
                    of = eval_point(ox)
                    if of is None:
                        break
                    ox, of = local_search(ox, of, max_steps=3)

        return X
