# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Grid-based decomposition with local Nelder-Mead simplex per cell."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        # We will collect all evaluated points and their objective values
        all_X = []
        all_F = []
        eval_count = [0]

        def eval_point(x):
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float)
            all_X.append(x.copy())
            all_F.append(f.copy())
            return f

        def count_eval(x):
            if eval_count[0] >= budget:
                raise StopIteration
            eval_count[0] += 1
            return eval_point(x)

        # Strategy: Use 4x4x4x4 grid = 256 cells, but only optimize the best ones
        # Actually, let's use a smarter approach:
        # 1. First, do a coarse grid evaluation to understand the landscape
        # 2. Then focus optimization on promising regions with different weights
        
        # Phase 1: Coarse grid sampling - 4^4 = 256 points is too many
        # Let's use 2^4 = 16 points for initial sampling
        # Then use remaining budget for local optimization
        
        # Better strategy: 
        # - Use 16 weight vectors (w from 0 to 1)
        # - For each weight, run Nelder-Mead from multiple starting points
        # - Distribute budget: 16 weights * 12 evals each = 192, leaving 8 for refinement
        
        weights = np.linspace(0.05, 0.95, 16)
        
        # For each weight, run Nelder-Mead from the center of the domain
        # with tight constraints to stay within bounds
        
        for w in weights:
            if eval_count[0] >= budget:
                break
            
            def scalar_obj(x, w=w):
                f = count_eval(x)
                return w * f[0] + (1 - w) * f[1]
            
            # Start from center of the domain
            x0 = (lo + hi) / 2.0
            
            # Run Nelder-Mead
            try:
                result = minimize(
                    scalar_obj,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 30,
                        'maxfev': 12,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
            except (StopIteration, Exception):
                pass
        
        # Phase 2: Use remaining budget to refine the best solutions found
        if eval_count[0] < budget and len(all_F) > 0:
            remaining = budget - eval_count[0]
            
            # Find the best points from Phase 1
            F_arr = np.array(all_F)
            
            # Get unique non-dominated points (simple filter)
            # For 2 objectives, a point is non-dominated if no other point is better in both
            n = len(F_arr)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if F_arr[j, 0] <= F_arr[i, 0] and F_arr[j, 1] <= F_arr[i, 1] and \
                       (F_arr[j, 0] < F_arr[i, 0] or F_arr[j, 1] < F_arr[i, 1]):
                        dominated[i] = True
                        break
            
            # Get non-dominated points
            nd_idx = [i for i in range(n) if not dominated[i]]
            
            # Sort by first objective
            nd_idx.sort(key=lambda i: F_arr[i, 0])
            
            # Pick up to 4 best non-dominated points as starting points for refinement
            n_refine = min(4, len(nd_idx), remaining // 4)
            
            for idx in range(n_refine):
                if eval_count[0] >= budget:
                    break
                
                start_x = all_X[nd_idx[idx]].copy()
                # Interpolate between adjacent non-dominated points to get diverse weights
                if idx > 0 and idx < len(nd_idx) - 1:
                    w = 0.5
                elif idx == 0:
                    w = 0.2
                else:
                    w = 0.8
                
                def refine_obj(x, w=w):
                    f = count_eval(x)
                    return w * f[0] + (1 - w) * f[1]
                
                try:
                    result = minimize(
                        refine_obj,
                        start_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 20,
                            'maxfev': max(2, remaining // max(1, n_refine - idx)),
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                except (StopIteration, Exception):
                    pass
        
        return all_X
# EVOLVE-BLOCK-END
