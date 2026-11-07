import numpy as np
import time
from scipy.optimize import minimize

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
        span = hi - lo
        used = 0
        
        archive_X = []
        archive_F = []

        def is_dominated(f, archive):
            for af in archive:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False

        def update_archive(x, f):
            nonlocal archive_X, archive_F
            new_X = []
            new_F = []
            for ax, af in zip(archive_X, archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_X.append(ax)
                    new_F.append(af)
            if not is_dominated(f, new_F):
                new_X.append(x)
                new_F.append(f)
            archive_X = new_X
            archive_F = new_F

        def evaluate(x):
            nonlocal used
            if used >= self.budget or time.time() > deadline - 5:
                return None
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            used += 1
            return x_clipped, f

        # Generate weight vectors
        n_grid = 4
        weights = []
        for i in range(n_grid + 1):
            for j in range(n_grid + 1 - i):
                k = n_grid - i - j
                w = np.array([i, j, k], dtype=float) / n_grid
                weights.append(w)
        
        # Add key points
        weights.append(np.array([1.0, 0.0, 0.0]))
        weights.append(np.array([0.0, 1.0, 0.0]))
        weights.append(np.array([0.0, 0.0, 1.0]))
        weights.append(np.array([0.5, 0.5, 0.0]))
        weights.append(np.array([0.5, 0.0, 0.5]))
        weights.append(np.array([0.0, 0.5, 0.5]))

        rng = np.random.default_rng(0)
        
        # Sort weights by some heuristic? Or just iterate.
        # Let's iterate and run local search.
        
        for w in weights:
            if used >= self.budget or time.time() > deadline - 10:
                break
            
            # Run local search from a random start
            x0 = lo + rng.random(self.dim) * span
            
            def local_obj(x):
                nonlocal used
                if used >= self.budget or time.time() > deadline - 5:
                    return np.inf
                x_clipped = np.clip(x, lo, hi)
                f = problem(x_clipped)
                used += 1
                return float(np.dot(w, f))
            
            try:
                # Use Nelder-Mead
                res = minimize(local_obj, x0, method='Nelder-Mead', 
                               options={'maxiter': 20, 'xatol': 1e-4, 'fatol': 1e-6})
                x_best = np.clip(res.x, lo, hi)
                f_best = problem(x_best)
                update_archive(x_best, f_best)
            except Exception:
                pass

        # Fill remaining budget with random sampling
        while used < self.budget and time.time() < deadline - 1:
            x = lo + rng.random(self.dim) * span
            res = evaluate(x)
            if res is None:
                break
            x, f = res
            update_archive(x, f)

        return np.array(archive_X) if archive_X else np.array([lo + span * 0.5])
