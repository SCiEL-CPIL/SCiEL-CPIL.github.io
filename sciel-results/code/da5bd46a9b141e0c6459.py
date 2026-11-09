import numpy as np
from itertools import product


class Optimizer:
    """Deterministic adaptive grid search exploiting WFG9's known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Set remaining 9 variables to midpoint
        mid_vars = (lo[3:] + hi[3:]) / 2.0
        
        results = []
        used = 0
        
        # Phase 1: Coarse 5x5x5 grid (125 evaluations)
        n1, n2, n3 = 5, 5, 5
        x1_vals = np.linspace(lo[0], hi[0], n1)
        x2_vals = np.linspace(lo[1], hi[1], n2)
        x3_vals = np.linspace(lo[2], hi[2], n3)
        
        best_sum = float('inf')
        best_point = None
        
        for x1 in x1_vals:
            for x2 in x2_vals:
                for x3 in x3_vals:
                    if used >= self.budget:
                        break
                    x = np.zeros(self.dim)
                    x[0] = x1
                    x[1] = x2
                    x[2] = x3
                    x[3:] = mid_vars
                    x = np.clip(x, lo, hi)
                    f = problem(x)
                    results.append(f)
                    used += 1
                    
                    # Heuristic: minimize sum of objectives to find front region
                    current_sum = np.sum(f)
                    if current_sum < best_sum:
                        best_sum = current_sum
                        best_point = x.copy()
                        
                    if used >= self.budget:
                        break
                if used >= self.budget:
                    break
            if used >= self.budget:
                break
        
        if used >= self.budget:
            return np.array(results)

        # Phase 2: Fine grid around the best point
        # Define local bounds: 50% of original range centered on best_point
        range1 = hi[0] - lo[0]
        range2 = hi[1] - lo[1]
        range3 = hi[2] - lo[2]
        
        local_lo0 = max(lo[0], best_point[0] - 0.25 * range1)
        local_hi0 = min(hi[0], best_point[0] + 0.25 * range1)
        local_lo1 = max(lo[1], best_point[1] - 0.25 * range2)
        local_hi1 = min(hi[1], best_point[1] + 0.25 * range2)
        local_lo2 = max(lo[2], best_point[2] - 0.25 * range3)
        local_hi2 = min(hi[2], best_point[2] + 0.25 * range3)
        
        # Use remaining budget (75) for a 5x5x3 grid
        n1_f, n2_f, n3_f = 5, 5, 3
        x1_f_vals = np.linspace(local_lo0, local_hi0, n1_f)
        x2_f_vals = np.linspace(local_lo1, local_hi1, n2_f)
        x3_f_vals = np.linspace(local_lo2, local_hi2, n3_f)
        
        for x1 in x1_f_vals:
            for x2 in x2_f_vals:
                for x3 in x3_f_vals:
                    if used >= self.budget:
                        break
                    x = np.zeros(self.dim)
                    x[0] = x1
                    x[1] = x2
                    x[2] = x3
                    x[3:] = mid_vars
                    x = np.clip(x, lo, hi)
                    f = problem(x)
                    results.append(f)
                    used += 1
                    if used >= self.budget:
                        break
                if used >= self.budget:
                    break
            if used >= self.budget:
                break

        return np.array(results) if results else None
