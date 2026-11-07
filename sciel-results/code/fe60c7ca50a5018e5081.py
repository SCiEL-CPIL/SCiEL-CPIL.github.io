import numpy as np
import time
from collections import deque

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls = 0
        history = []

        def try_eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return None
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            history.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def line_search_1d(x_start, direction, f_start):
            """
            1D minimization along direction.
            Returns (best_x, best_f, num_evals)
            """
            # Try alpha = 1.0
            x1 = x_start + 1.0 * direction
            f1 = try_eval(x1)
            if f1 is None:
                return x_start, f_start, 1
            
            if f1 < f_start:
                # Try alpha = 2.0
                x2 = x_start + 2.0 * direction
                f2 = try_eval(x2)
                if f2 is None:
                    return x1, f1, 2
                
                if f2 < f1:
                    return x2, f2, 2
                else:
                    # Try alpha = 0.5
                    x05 = x_start + 0.5 * direction
                    f05 = try_eval(x05)
                    if f05 is None:
                        return x1, f1, 3
                    if f05 < f1:
                        return x05, f05, 3
                    else:
                        return x1, f1, 3
            else:
                # Try alpha = 0.5
                x05 = x_start + 0.5 * direction
                f05 = try_eval(x05)
                if f05 is None:
                    return x_start, f_start, 2
                
                if f05 < f_start:
                    return x05, f05, 2
                else:
                    # Try alpha = -0.5
                    x_neg = x_start - 0.5 * direction
                    f_neg = try_eval(x_neg)
                    if f_neg is None:
                        return x_start, f_start, 3
                    if f_neg < f_start:
                        return x_neg, f_neg, 3
                    else:
                        return x_start, f_start, 3

        def powell_method(x0, max_calls):
            nonlocal calls, best_x, best_f
            if max_calls < 10:
                return x0, best_f

            x_current = x0.copy()
            f_current = try_eval(x_current)
            if f_current is None:
                return x0, best_f

            directions = [np.eye(n)[i] for i in range(n)]
            
            while calls < budget and time.time() < deadline - 5:
                x_old = x_current.copy()
                f_old = f_current
                min_improvement = float('inf')
                min_dir_idx = -1
                
                for i in range(n):
                    if calls >= budget or time.time() > deadline - 5:
                        break
                    
                    d = directions[i]
                    x_new, f_new, _ = line_search_1d(x_current, d, f_current)
                    
                    improvement = f_current - f_new
                    if improvement > min_improvement:
                        min_improvement = improvement
                        min_dir_idx = i
                    
                    x_current = x_new
                    f_current = f_new
                
                if np.linalg.norm(x_current - x_old) < 1e-6 and abs(f_current - f_old) < 1e-6:
                    break
                
                if min_dir_idx != -1 and min_improvement > 1e-8:
                    new_dir = x_current - x_old
                    if np.linalg.norm(new_dir) > 1e-8:
                        directions[min_dir_idx] = new_dir

            return x_current, best_f

        # Phase 1: Global Sampling
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        num_samples = 60
        for i in range(1, num_samples + 1):
            point = []
            for j in range(n):
                point.append(halton(i, bases[j]) * 10.0 - 5.0)
            try_eval(np.array(point))
            if calls >= 80 or time.time() > deadline - 10:
                break
        
        # Add center
        try_eval(np.zeros(n))
        
        # Add random
        for _ in range(20):
            if calls >= 80 or time.time() > deadline - 10:
                break
            try_eval(rng.uniform(-5, 5, n))

        # Phase 2: Local Optimization
        if history:
            # Sort by function value
            history.sort(key=lambda t: t[0])
            starts = []
            for f, x in history:
                is_dup = False
                for s in starts:
                    if np.linalg.norm(x - s) < 0.5:
                        is_dup = True
                        break
                if not is_dup:
                    starts.append(x)
                if len(starts) == 4:
                    break

            remaining_budget = budget - calls
            if remaining_budget > 0 and starts:
                budget_per_start = max(10, remaining_budget // len(starts))
                for x_start in starts:
                    if calls >= budget - 5 or time.time() > deadline - 5:
                        break
                    remaining = budget - calls
                    alloc = min(budget_per_start, remaining)
                    if alloc < 10:
                        break
                    powell_method(x_start, alloc)

        return best_x, best_f
