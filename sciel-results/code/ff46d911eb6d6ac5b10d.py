import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        max_calls = self.budget
        calls_made = 0
        dim = self.dim
        best_x = None
        best_f = np.inf

        def eval_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= max_calls:
                return np.inf
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Multi-start strategy: try several starting points, each followed by L-BFGS-B
        rng = np.random.default_rng(42)
        
        # Generate diverse starting points
        starts = [np.zeros(dim)]
        # Add random points in the box
        for _ in range(4):
            starts.append(rng.uniform(-4.0, 4.0, size=dim))
        # Add points along axes
        for i in range(dim):
            pt = np.zeros(dim)
            pt[i] = 3.0
            starts.append(pt)
            pt2 = np.zeros(dim)
            pt2[i] = -3.0
            starts.append(pt2)

        # Shuffle starts but keep origin first
        starts = [starts[0]] + starts[1:]
        rng.shuffle(starts[1:])

        # Allocate budget: ~15% per start for exploration, rest for polish
        # Use 5 starts, give each a budget slice
        num_starts = min(len(starts), 6)
        starts = starts[:num_starts]
        
        # Try each start with L-BFGS-B
        for idx, start in enumerate(starts):
            if time.time() > deadline:
                break
            if calls_made >= max_calls:
                break
            
            # Give each start a portion of remaining budget
            remaining = max_calls - calls_made
            if remaining < 10:
                break
            
            # For early starts, give more budget for exploration
            if idx == 0:
                alloc = int(remaining * 0.4)
            elif idx == 1:
                alloc = int(remaining * 0.25)
            else:
                alloc = int(remaining * 0.15)
            
            alloc = max(alloc, 5)
            alloc = min(alloc, remaining)
            
            nm_calls = 0
            nm_limit = alloc

            def lbfgs_func(z):
                nonlocal nm_calls
                if nm_calls >= nm_limit:
                    return np.inf
                nm_calls += 1
                return func(np.clip(z, -5.0, 5.0))

            try:
                res = minimize(lbfgs_func, start, method='L-BFGS-B',
                               bounds=[(-5.0, 5.0)] * dim,
                               options={'maxiter': max(1, nm_limit // 2),
                                        'maxfun': nm_limit,
                                        'ftol': 1e-12,
                                        'gtol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        # If we still have budget, do a final polish from best point
        if best_x is not None and calls_made < max_calls and time.time() < deadline:
            remaining = max_calls - calls_made
            if remaining > 10:
                nm_calls = 0
                nm_limit = remaining

                def lbfgs_func2(z):
                    nonlocal nm_calls
                    if nm_calls >= nm_limit:
                        return np.inf
                    nm_calls += 1
                    return func(np.clip(z, -5.0, 5.0))

                try:
                    res = minimize(lbfgs_func2, best_x, method='L-BFGS-B',
                                   bounds=[(-5.0, 5.0)] * dim,
                                   options={'maxiter': max(1, nm_limit // 2),
                                            'maxfun': nm_limit,
                                            'ftol': 1e-15,
                                            'gtol': 1e-10})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5.0, 5.0)
                except Exception:
                    pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = eval_func(best_x)

        return best_x, best_f
