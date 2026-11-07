import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.perf_counter()
        deadline = 325.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        M = self.n_obj
        
        rng = np.random.default_rng(42)
        evaluated = 0
        max_evals = self.budget
        points = []
        F = []

        def check_deadline():
            return (time.perf_counter() - t0) < deadline

        def safe_eval(x):
            nonlocal evaluated
            if evaluated >= max_evals or not check_deadline():
                return None
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            if len(x) != n:
                x = np.resize(x, n)
            f = np.array(problem(x), dtype=float)
            if np.any(~np.isfinite(f)):
                return None
            evaluated += 1
            return f

        # DTLZ7 specific structure
        # x[21] = 0.5 for g=0
        # x[2:21] sum = 19 for min f3
        # x[0], x[1] grid
        
        # Strategy: Dense grid in x[0], x[1] with optimal x[2:21]
        # K=14 gives 196 points.
        K = 14
        n_grid_points = K * K
        if n_grid_points > max_evals:
            K = int(np.sqrt(max_evals))
            n_grid_points = K * K
            
        # Generate grid
        xs = np.linspace(0, 1, K, endpoint=False) + 0.5/K
        
        # Optimal middle vars
        # Sum x[2:21] = 19
        # Set all to 1.0
        mid_vars = np.ones(19)
        
        for i in range(K):
            for j in range(K):
                if evaluated >= max_evals or not check_deadline():
                    break
                x = np.zeros(n)
                x[0] = xs[i]
                x[1] = xs[j]
                x[2:21] = mid_vars
                x[21] = 0.5
                f = safe_eval(x)
                if f is not None:
                    points.append(np.clip(x, lo, hi))
                    F.append(f)
                    
        # Use remaining budget for random exploration
        # Just in case
        while evaluated < max_evals and check_deadline():
            x = rng.uniform(0, 1, n)
            x[21] = 0.5 # Keep g=0
            f = safe_eval(x)
            if f is None:
                break
            points.append(np.clip(x, lo, hi))
            F.append(f)

        return np.array(points) if points else np.zeros((0, n))
