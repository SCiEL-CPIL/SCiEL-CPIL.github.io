import numpy as np
import random

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        evals = 0
        all_points = []
        all_f = []
        
        def evaluate(x):
            nonlocal evals
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            evals += 1
            all_points.append(x_clipped)
            all_f.append(f)
            return f

        # Phase 1: Random Exploration
        n_explore = self.budget // 2
        for _ in range(n_explore):
            x = lo + rng.random(self.dim) * (hi - lo)
            evaluate(x)
            
        # Phase 2: Identify Non-Dominated Set
        F = np.array(all_f)
        X = np.array(all_points)
        
        def non_dominated_sort(F):
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        nd_idx = non_dominated_sort(F)
        
        if len(nd_idx) == 0:
            return all_points
            
        # Phase 3: Local Search on Representative Points
        # Select K diverse points from the non-dominated set
        K = 10
        if len(nd_idx) > K:
            # Simple diversity selection: pick points with max crowding distance
            # Or just pick evenly spaced indices in the sorted non-dominated set
            # Let's use crowding distance
            cd = np.zeros(len(nd_idx))
            for m in range(self.n_obj):
                sorted_idx = np.argsort(F[nd_idx, m])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for k in range(1, len(sorted_idx) - 1):
                    f_range = F[nd_idx[sorted_idx[-1]], m] - F[nd_idx[sorted_idx[0]], m]
                    if f_range > 0:
                        cd[sorted_idx[k]] += (
                            F[nd_idx[sorted_idx[k + 1]], m] - F[nd_idx[sorted_idx[k - 1]], m]
                        ) / f_range
            # Pick top K by crowding distance
            selected_local_nd = nd_idx[np.argsort(cd)[-K:]]
        else:
            selected_local_nd = nd_idx
            
        # Ideal point (min observed)
        ideal_point = np.min(F, axis=0)
        
        # Local Search: Nelder-Mead minimizing Chebyshev distance to ideal point
        def objective(x):
            f = problem(np.clip(x, lo, hi))
            evals_local[0] += 1
            # Chebyshev distance to ideal point
            dist = np.max(np.abs(f - ideal_point))
            return dist

        # We need to track evaluations carefully to not exceed budget
        # The `evaluate` function above increments `evals`.
        # We need a local evaluator that increments `evals` but doesn't add to `all_points` until the end?
        # Actually, we should record all points.
        
        # Let's modify the strategy:
        # The local search will call `problem` directly.
        # We must ensure total evals <= budget.
        
        remaining_budget = self.budget - evals
        evals_per_search = max(1, remaining_budget // len(selected_local_nd))
        
        for idx in selected_local_nd:
            x0 = X[idx]
            # Run Nelder-Mead
            # We need to wrap the objective to count evals
            evals_local = [0]
            
            def local_obj(x):
                if evals_local[0] >= evals_per_search:
                    return np.inf
                fx = problem(np.clip(x, lo, hi))
                evals_local[0] += 1
                evals += 1
                all_points.append(np.clip(x, lo, hi))
                all_f.append(fx)
                dist = np.max(np.abs(fx - ideal_point))
                return dist

            # Simple Nelder-Mead implementation
            # Or use scipy? SciPy is allowed.
            # But we need to control evals.
            # Let's use a simple coordinate descent or Nelder-Mead.
            
            # Using SciPy's minimize with method='Nelder-Mead'
            # We can't easily limit function calls in SciPy's minimize without a wrapper that raises an exception or returns inf.
            # Returning inf stops the improvement but might still count evals.
            
            from scipy.optimize import minimize
            
            res = minimize(local_obj, x0, method='Nelder-Mead', 
                           options={'maxiter': evals_per_search, 'xatol': 1e-4, 'fatol': 1e-4})
            
            # Note: minimize might call the function more times than maxiter?
            # maxiter limits the number of iterations, but each iteration can have multiple function calls.
            # To be safe, we rely on the evals_local check inside local_obj.
            
        return all_points
