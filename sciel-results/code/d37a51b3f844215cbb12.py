import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
            dim = self.dim
            n_obj = self.n_obj
            budget = self.budget
            
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            mid = (lo + hi) / 2.0
            span = hi - lo
            
            all_x = []
            all_f = []
            eval_count = 0
            
            def evaluate(x):
                nonlocal eval_count
                if eval_count >= budget:
                    return None
                f = problem(x)
                eval_count += 1
                all_x.append(x.copy())
                all_f.append(np.array(f))
                return f

            # 1. Initial random sampling to estimate bounds of objectives
            n_init = 10
            for _ in range(n_init):
                x = lo + span * np.random.rand(dim)
                evaluate(x)
                
            if not all_f:
                return np.array([mid])
                
            f_array = np.array(all_f)
            z_min = f_array.min(axis=0)
            z_max = f_array.max(axis=0)
            z_range = z_max - z_min + 1e-10

            # 2. Generate 30 weight vectors
            n_weights = 30
            rng = np.random.default_rng(42)
            weights = []
            for _ in range(n_weights):
                w = rng.dirichlet(np.ones(n_obj))
                weights.append(w)
            weights = np.array(weights)
            
            # 3. Allocate remaining budget
            remaining_budget = budget - eval_count
            maxfev_per_weight = max(1, remaining_budget // n_weights)
            
            # 4. Run Nelder-Mead for each weight
            for w in weights:
                if eval_count >= budget:
                    break
                
                # Define the scalarization function for scipy
                def scalar_obj(x, w=w):
                    if eval_count >= budget:
                        return 1e10
                    # Clip x to bounds to be safe, though scipy might step out
                    x_clipped = np.clip(x, lo, hi)
                    # We need to use the evaluate function to count, but scipy expects a float return
                    # We can't easily return None. 
                    # If we exceed budget, we return a large penalty.
                    # But we must ensure we don't double count if problem is called.
                    # The evaluate function above increments eval_count.
                    # We should call evaluate inside scalar_obj.
                    
                    # Check if this specific x was already evaluated? 
                    # Nelder-Mead might re-evaluate. 
                    # For simplicity, we assume distinct evaluations.
                    
                    f = evaluate(x_clipped)
                    if f is None:
                        return 1e10
                    # Tchebycheff scalarization
                    # Note: z_min/z_max are fixed from initial phase
                    return np.max(w * np.abs(f - z_min) / z_range)

                # Start from center
                x0 = mid.copy()
                
                # Use Nelder-Mead
                # bounds are handled by clipping in scalar_obj
                res = minimize(
                    scalar_obj, 
                    x0, 
                    method='Nelder-Mead', 
                    options={'maxfev': maxfev_per_weight, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                
                # The result x is in res.x, but we have already recorded all evaluations in all_x/all_f
                # We don't strictly need res.x for the final return, as we return non-dominated set of all recorded points.
            
            # 5. Find non-dominated points
            if not all_f:
                return np.array([mid])
                
            all_f = np.array(all_f)
            all_x = np.array(all_x)
            
            n_points = len(all_f)
            dominated = np.zeros(n_points, dtype=bool)
            for i in range(n_points):
                for j in range(n_points):
                    if i == j:
                        continue
                    if np.all(all_f[j] <= all_f[i]) and np.any(all_f[j] < all_f[i]):
                        dominated[i] = True
                        break
                    
            non_dominated_idx = np.where(~dominated)[0]
            return all_x[non_dominated_idx]
