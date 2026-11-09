import numpy as np
import time

def halton(index, base):
    f = 1.0
    r = 0.0
    i = index
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r

def halton_point(index, dim):
    point = np.zeros(dim)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for i in range(dim):
        point[i] = halton(index, primes[i])
    return point

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0
        history = []  # List of (f_val, x)

        def evaluate(x):
            nonlocal best_x, best_f, calls
            if calls >= budget or time.time() >= deadline:
                return
            x = np.clip(x, -5, 5)
            f_val = func(x)
            calls += 1
            history.append((f_val, x.copy()))
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()

        # Phase 1: Global Exploration with Halton Sequence
        global_budget = int(budget * 0.5) # 50% for global
        
        # Start with center
        evaluate(np.zeros(dim))
        
        idx = 1
        while calls < global_budget and time.time() < deadline - 5:
            point_unit = halton_point(idx, dim)
            point = point_unit * 10.0 - 5.0
            evaluate(point)
            idx += 1
            
        if calls >= budget or best_x is None:
            return best_x, best_f

        # Phase 2: Local Refinement of Top Candidates
        # Sort history by function value
        history.sort(key=lambda item: item[0])
        
        # Take top 3 candidates
        top_candidates = history[:3]
        
        # Allocate remaining budget to local search
        remaining_budget = budget - calls
        # Distribute budget among candidates, giving more to the best
        # e.g., 50% to best, 30% to 2nd, 20% to 3rd
        if len(top_candidates) >= 1:
            allocs = [int(remaining_budget * 0.5), int(remaining_budget * 0.3), int(remaining_budget * 0.2)]
            # Adjust if sum exceeds remaining
            total_alloc = sum(allocs)
            if total_alloc > remaining_budget:
                # Scale down
                factor = remaining_budget / total_alloc
                allocs = [int(a * factor) for a in allocs]
        else:
            allocs = [remaining_budget]

        for i, (f_init, x_init) in enumerate(top_candidates):
            if i >= len(allocs):
                break
            local_budget = allocs[i]
            if local_budget <= 0:
                continue
                
            # Run (1+1)-ES
            x = x_init.copy()
            fx = f_init
            sigma = 0.5
            
            no_improve = 0
            max_no_improve = 10
            
            while calls < budget and time.time() < deadline - 2:
                if no_improve > max_no_improve:
                    sigma *= 0.5
                    no_improve = 0
                    if sigma < 1e-5:
                        break
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.25
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    no_improve += 1

        return best_x, best_f
