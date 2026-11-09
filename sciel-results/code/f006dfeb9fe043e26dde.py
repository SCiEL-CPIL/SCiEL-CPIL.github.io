"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-objective optimizer exploiting WFG4's separability structure.
    
    WFG4: variables 0,1 are 'deciding' (control Pareto front position),
    variables 2-11 are 'positioning' (control distance from front).
    
    Strategy:
    1. Grid-sample deciding variables (2D grid) for Pareto front coverage.
    2. For each grid point, locally optimize positioning variables to reach front.
    3. Use remaining budget for diversity and refinement.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        eval_count = 0
        all_X = []
        all_F = []
        
        def safe_evaluate(x):
            nonlocal eval_count
            if eval_count >= budget or time.time() > deadline:
                return None, None
            x_c = np.clip(x, lo, hi)
            f = np.array(problem(x_c))
            eval_count += 1
            all_X.append(x_c)
            all_F.append(f)
            return f, x_c
        
        rng = np.random.default_rng(123)
        
        # Phase 1: Grid sample deciding variables (indices 0,1)
        # Use 6x6 = 36 grid points to leave budget for optimization
        n_grid = 6
        grid_deciding = []
        for i in range(n_grid):
            for j in range(n_grid):
                t0 = (i + 0.5) / n_grid
                t1 = (j + 0.5) / n_grid
                d0 = lo[0] + t0 * (hi[0] - lo[0])
                d1 = lo[1] + t1 * (hi[1] - lo[1])
                grid_deciding.append(np.array([d0, d1]))
        
        # Phase 2: For each grid point, initialize + locally optimize positioning vars
        for gp in grid_deciding:
            if eval_count >= budget or time.time() > deadline:
                break
            
            # Initialize positioning variables randomly
            pos_init = lo[2:] + (hi[2:] - lo[2:]) * rng.random(dim - 2)
            x_init = np.concatenate([gp, pos_init])
            f_init, x_init = safe_evaluate(x_init)
            if f_init is None:
                break
            
            # Local optimization: minimize sum of objectives over positioning vars
            def obj_local(pos_vars):
                x = np.concatenate([gp, pos_vars])
                x = np.clip(x, lo, hi)
                f = np.array(problem(x))
                return np.sum(f)
            
            # Track evals within this local opt
            evals_before = eval_count
            try:
                res = minimize(
                    obj_local,
                    pos_init,
                    method='Nelder-Mead',
                    options={'maxiter': 15, 'xatol': 1e-3, 'fatol': 1e-3}
                )
                # Evaluate the result
                x_res = np.concatenate([gp, res.x])
                f_res, x_res = safe_evaluate(x_res)
            except Exception:
                pass
        
        # Phase 3: Use remaining budget for random diversity
        while eval_count < budget and time.time() < deadline:
            x = lo + (hi - lo) * rng.random(dim)
            safe_evaluate(x)
        
        # Phase 4: Local refinement around promising points
        if len(all_F) > 0 and eval_count < budget and time.time() < deadline:
            all_F_arr = np.array(all_F)
            all_X_arr = np.array(all_X)
            
            # Find non-dominated points
            def is_dom(f, F):
                return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))
            
            nd_mask = ~np.array([is_dom(f, all_F_arr) for f in all_F_arr])
            nd_X = all_X_arr[nd_mask]
            
            # Refine top few non-dominated points
            for i in range(min(10, len(nd_X))):
                if eval_count >= budget or time.time() > deadline:
                    break
                
                x0 = nd_X[i].copy()
                
                def obj_refine(x):
                    x = np.clip(x, lo, hi)
                    f = np.array(problem(x))
                    return np.sum(f)
                
                try:
                    res = minimize(
                        obj_refine,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 10, 'xatol': 1e-4, 'fatol': 1e-4}
                    )
                    x_ref = np.clip(res.x, lo, hi)
                    safe_evaluate(x_ref)
                except Exception:
                    pass
        
        return np.array(all_X) if all_X else (lo + hi) / 2.0
# EVOLVE-BLOCK-END
