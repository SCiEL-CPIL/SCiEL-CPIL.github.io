import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, aim for 290s
        deadline = start_time + 285.0
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        eval_count = 0
        history = []

        def counting_func(x):
            nonlocal eval_count, best_f, best_x
            if eval_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            try:
                x_arr = np.asarray(x, dtype=float)
                if x_arr.shape != (dim,):
                    x_arr = x_arr.reshape(dim)
                x_clipped = np.clip(x_arr, -5, 5)
                val = float(func(x_clipped))
                eval_count += 1
                history.append((val, x_clipped.copy()))
                if val < best_f:
                    best_f = val
                    best_x = x_clipped.copy()
                return val
            except Exception:
                eval_count += 1
                return np.inf

        # Phase 1: Sparse Global Sampling (Halton)
        # Provides a good initial spread and a baseline best.
        def halton_sequence(n, dim):
            points = np.zeros((n, dim))
            for i in range(1, n + 1):
                for d in range(dim):
                    base = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][d % 10]
                    f = 1.0
                    r = 0.0
                    i_val = i
                    while i_val > 0:
                        f /= base
                        r += f * (i_val % base)
                        i_val //= base
                    points[i-1, d] = r
            return points

        n_initial = 30
        halton_pts = halton_sequence(n_initial, dim)
        halton_pts = halton_pts * 10.0 - 5.0
        
        for pt in halton_pts:
            if eval_count >= budget or time.time() > deadline:
                break
            counting_func(pt)

        # Phase 2: Differential Evolution for Global Optimization
        # DE is robust for multimodal functions like Katsuura.
        # We allocate a significant portion of the budget here.
        # popsize * maxiter should be less than remaining budget.
        remaining = budget - eval_count
        if remaining > 50 and best_x is not None:
            popsize = 15
            # Estimate maxiter to fit within remaining budget (leave some for polish)
            # We want to use ~80% of remaining for DE
            de_budget = int(remaining * 0.8)
            maxiter_de = max(1, de_budget // popsize)
            
            try:
                # bounds is [-5, 5] for all dimensions
                bounds = [(-5.0, 5.0)] * dim
                
                # Use a fixed seed for reproducibility if needed, but random is fine
                # tol and xatol can be set, but maxiter is the main control
                res_de = scipy.optimize.differential_evolution(
                    counting_func,
                    bounds,
                    maxiter=maxiter_de,
                    popsize=popsize,
                    tol=0.01,
                    xatol=1e-5,
                    seed=42,
                    init='latinhypercube' # Better initial spread than random
                )
                
                if res_de.fun < best_f:
                    best_f = res_de.fun
                    best_x = res_de.x
            except Exception:
                pass

        # Phase 3: Local Polish using Nelder-Mead
        # Katsuura has a sharp minimum. DE might land in the right basin but not at the exact bottom.
        # NM with tight tolerances can exploit this.
        remaining = budget - eval_count
        if remaining > 10 and best_x is not None:
            try:
                # Use a small initial simplex to avoid jumping out of the local basin
                # NM default simplex is based on x0. We can try to control it.
                # scipy's NM doesn't have an explicit initial_simplex param in all versions,
                # but it uses x0 + delta * identity.
                # We can just rely on maxiter and tolerances.
                
                res_nm = scipy.optimize.minimize(
                    counting_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining,
                        'xatol': 1e-12,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = res_nm.x
            except Exception:
                pass

        # Fallback if no valid result
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
                eval_count += 1
            except Exception:
                best_f = 0.0

        return best_x, best_f
