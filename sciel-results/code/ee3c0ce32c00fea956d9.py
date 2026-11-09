import numpy as np
import time
from scipy.optimize import differential_evolution

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

        ref_point = np.array([3000.0, 0.06])

        X_all = []
        F_all = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if time.time() > deadline:
                return None
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            X_all.append(x)
            F_all.append(f)
            eval_count += 1
            return f

        # Phase 1: Sobol initialization
        n_init = 30
        # Use simple random for Sobol-like behavior since scipy.stats.qmc might not be available in all contexts,
        # but the task says SciPy is available. Let's use scipy.stats.qmc if possible, else random.
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True)
            X_init = sampler.random(n_init)
        except ImportError:
            X_init = np.random.rand(n_init, dim)
        
        X_init = lo + (hi - lo) * X_init
        for x in X_init:
            f = evaluate(x)
            if f is None:
                break

        if time.time() > deadline:
            return X_all

        # Phase 2: Weighted sum optimization with Differential Evolution
        # We have remaining budget. Let's use a fraction for DE.
        # DE requires multiple evaluations per run.
        # Let's do a few weighted-sum optimizations.
        
        # Remaining budget
        rem_budget = budget - eval_count
        if rem_budget <= 0:
            return X_all

        # Number of weighted sums to try
        # Each DE run should be limited.
        # Let's say we do 4 weighted sum optimizations.
        # Each DE run gets rem_budget // 4 evaluations.
        n_ws = 4
        budget_per_ws = max(10, rem_budget // n_ws)
        
        # Weights for weighted sum: (1,0), (0,1), (0.5,0.5), (0.2,0.8), (0.8,0.2) etc.
        # Let's use a set of well-spread weights.
        weights = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.5, 0.5]),
            np.array([0.2, 0.8]),
            np.array([0.8, 0.2]),
            np.array([0.1, 0.9]),
            np.array([0.9, 0.1]),
        ]
        
        # Shuffle weights to randomize order? Or just take first n_ws.
        # Let's take a diverse subset.
        ws_indices = np.random.choice(len(weights), size=min(n_ws, len(weights)), replace=False)
        
        for w_idx in ws_indices:
            if time.time() > deadline:
                break
            if eval_count >= budget:
                break
                
            w = weights[w_idx]
            
            def obj_func(x):
                # If we run out of budget or time, return a large value to stop
                if time.time() > deadline or eval_count >= budget:
                    return 1e10
                f = evaluate(x)
                if f is None:
                    return 1e10
                return np.dot(f, w)
            
            try:
                # Differential Evolution
                # popsize=15, tol=1e-4, maxiter=budget_per_ws//15, polish=False
                # maxfevs is the key parameter
                result = differential_evolution(
                    obj_func, 
                    bounds=list(zip(lo, hi)),
                    maxiter=100,
                    popsize=15,
                    tol=1e-4,
                    seed=42,
                    polish=False,
                    maxfevs=budget_per_ws
                )
            except Exception:
                pass

        # Phase 3: Local refinement using Nelder-Mead on the best points found
        # Select top K non-dominated points
        F_arr = np.array(F_all) if F_all else np.empty((0, n_obj))
        X_arr = np.array(X_all) if X_all else np.empty((0, dim))
        
        if len(F_arr) > 0:
            # Simple non-dominated selection: pick points that are not dominated by any other
            # Or just pick the best few based on some metric
            # Let's pick points with low sum of objectives
            sums = np.sum(F_arr, axis=1)
            # Sort by sum
            sorted_idx = np.argsort(sums)
            # Take top 5 candidates for local refinement
            candidates = sorted_idx[:5]
            
            rem_budget = budget - eval_count
            if rem_budget > 0:
                budget_per_local = max(5, rem_budget // len(candidates))
                
                for idx in candidates:
                    if time.time() > deadline:
                        break
                    if eval_count >= budget:
                        break
                    
                    x0 = X_arr[idx]
                    
                    def local_obj(x):
                        if time.time() > deadline or eval_count >= budget:
                            return 1e10
                        f = evaluate(x)
                        if f is None:
                            return 1e10
                        # Use a weighted sum with weights based on the point's direction?
                        # Or just minimize the first objective?
                        # Better: minimize the distance to the reference point in normalized space?
                        # Or just use a weighted sum with weights proportional to 1/f?
                        # Let's try minimizing the sum again, or a specific weight.
                        # Since we don't know the ideal weights, let's just do a small DE or NM on the sum.
                        return np.sum(f)
                    
                    try:
                        from scipy.optimize import minimize
                        # Nelder-Mead
                        res = minimize(local_obj, x0, method='Nelder-Mead', 
                                       options={'maxiter': budget_per_local, 'xatol': 1e-6, 'fatol': 1e-6})
                    except Exception:
                        pass

        return X_all
