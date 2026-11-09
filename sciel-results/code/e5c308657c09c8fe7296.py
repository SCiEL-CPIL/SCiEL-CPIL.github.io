# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """LHS sampling + Nelder-Mead polish for WFG7 multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        deadline = time.time() + 280.0  # safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        X = []
        F = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= budget or time.time() > deadline:
                return None
            fx = problem(x)
            eval_count += 1
            X.append(x)
            F.append(fx)
            return fx

        # Phase 1: LHS sampling for broad coverage
        n_lhs = 48  # 48 points in 12 dims = 4 per dimension
        # Latin Hypercube: each dim divided into n_lhs equal intervals, one point per interval
        lhs_points = []
        for i in range(n_lhs):
            point = np.zeros(dim)
            for d in range(dim):
                # In interval i for dimension d, pick random position within [i/(n), (i+1)/n)
                u = np.random.rand()
                frac = (i + u) / n_lhs
                point[d] = lo[d] + frac * span[d]
            lhs_points.append(point)
        
        for x in lhs_points:
            evaluate(x)
            if eval_count >= budget:
                break

        # Phase 2: Find non-dominated points and polish with Nelder-Mead
        if len(F) > 2:
            F_arr = np.array(F)
            
            # Simple non-dominated filter
            n = len(F_arr)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(i + 1, n):
                    if dominated[j]:
                        continue
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dominated[j] = True
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break
            
            # Get non-dominated indices
            nd_indices = [i for i in range(n) if not dominated[i]]
            
            # Polish top non-dominated points with Nelder-Mead
            # Use hypervolume-like heuristic: prioritize points that are extreme in some objective
            # Sort by sum of objectives (lower is better for minimization)
            nd_indices_sorted = sorted(nd_indices, key=lambda i: F_arr[i].sum())
            
            # Polish up to 8 best non-dominated points
            n_polish = min(8, len(nd_indices_sorted), budget - eval_count)
            
            for idx in nd_indices_sorted[:n_polish]:
                if time.time() > deadline or eval_count >= budget:
                    break
                
                x0 = X[idx].copy()
                
                # Nelder-Mead on sum of objectives (scalar surrogate)
                def objective(x):
                    if eval_count >= budget:
                        return np.inf
                    fx = problem(x)
                    eval_count += 1
                    X.append(x.copy())
                    F.append(fx)
                    return float(np.sum(fx))
                
                try:
                    res = minimize(objective, x0, method='Nelder-Mead',
                                   options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-4,
                                            'adaptive': True})
                except Exception:
                    pass
                
                if time.time() > deadline:
                    break

        return X
# EVOLVE-BLOCK-END
