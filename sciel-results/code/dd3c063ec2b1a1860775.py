import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        eval_count = 0
        
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
                if val < best_f:
                    best_f = val
                    best_x = x_clipped.copy()
                return val
            except Exception:
                eval_count += 1
                return np.inf
        
        # Phase 1: Structured initial sampling (Halton sequence)
        # Use 30 points for good coverage without wasting too much budget
        n_initial = 30
        
        def halton_sequence(n, dim):
            """Generate n points in dim dimensions using Halton sequence."""
            points = []
            for i in range(1, n + 1):
                pt = []
                for d in range(dim):
                    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
                    base = bases[d % len(bases)]
                    f = 1.0
                    r = 0.0
                    idx = i
                    while idx > 0:
                        f /= base
                        r += f * (idx % base)
                        idx //= base
                    pt.append(r)
                points.append(pt)
            return np.array(points)
        
        # Generate initial points
        init_points = halton_sequence(n_initial, dim)
        # Map from [0,1]^dim to [-5,5]^dim
        init_points = init_points * 10.0 - 5.0
        
        # Evaluate all initial points
        for pt in init_points:
            if eval_count >= budget or time.time() > deadline:
                break
            counting_func(pt)
        
        # Phase 2: CMA-ES from the best point
        # Use moderate sigma and popsize for balance between exploration and exploitation
        remaining = budget - eval_count
        if remaining > 40 and best_x is not None:
            try:
                popsize = 12
                max_iter = max(5, remaining // popsize - 5)
                
                res = scipy.optimize.cma(
                    counting_func,
                    best_x,
                    sigma0=1.2,
                    options={
                        'maxiter': max_iter,
                        'popsize': popsize,
                        'tol': 1e-10,
                        'verbose': 0,
                        'bounds': [(-5.0, 5.0)] * dim
                    }
                )
            except Exception:
                pass
        
        # Phase 3: Multi-start Nelder-Mead for local refinement
        # Use top candidates from all evaluations so far
        remaining = budget - eval_count
        if remaining > 15 and best_x is not None:
            try:
                # Create multiple starting points around the best solution
                starts = [best_x.copy()]
                
                # Add a few perturbed versions
                rng = np.random.default_rng(42)
                for _ in range(3):
                    perturbation = rng.normal(0, 0.5, size=dim)
                    new_x = np.clip(best_x + perturbation, -5, 5)
                    starts.append(new_x)
                
                # Run Nelder-Mead from each start
                for start_pt in starts:
                    if eval_count >= budget or time.time() > deadline:
                        break
                    remaining = budget - eval_count
                    if remaining < 10:
                        break
                    try:
                        res_nm = scipy.optimize.minimize(
                            counting_func,
                            start_pt,
                            method='Nelder-Mead',
                            options={
                                'maxiter': remaining // 4,
                                'xatol': 1e-12,
                                'fatol': 1e-12,
                                'adaptive': True
                            }
                        )
                    except Exception:
                        pass
            except Exception:
                pass
        
        # Phase 4: Final high-precision polish
        remaining = budget - eval_count
        if remaining > 5 and best_x is not None:
            try:
                res_final = scipy.optimize.minimize(
                    counting_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining,
                        'xatol': 1e-14,
                        'fatol': 1e-14,
                        'adaptive': True
                    }
                )
            except Exception:
                pass
        
        # Ensure we have a valid result
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
                eval_count += 1
            except Exception:
                best_f = 0.0
        
        return best_x, best_f
