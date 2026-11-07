import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        d = self.dim
        budget = self.budget
        
        best_x = None
        best_f = float('inf')
        calls = 0
        
        def check_deadline():
            return time.time() > deadline or calls >= budget

        def nelder_mead(x0, budget_limit):
            nonlocal calls, best_x, best_f
            # Init simplex
            simplex = [x0.copy()]
            fs = [func(np.clip(x0, -5, 5))]
            calls += 1
            if fs[0] < best_f:
                best_f, best_x = fs[0], x0.copy()
                
            for i in range(d):
                xi = x0.copy()
                xi[i] += 0.25 if x0[i] != 0 else 0.05
                simplex.append(xi)
                fi = func(np.clip(xi, -5, 5))
                fs.append(fi)
                calls += 1
                if fi < best_f:
                    best_f, best_x = fi, xi.copy()
                    
            alpha, gamma, rho, sigma_ = 1.0, 2.0, 0.5, 0.5
            
            while calls < calls_start + budget_limit and not check_deadline():
                # Sort
                idx = np.argsort(fs)
                simplex = [simplex[i] for i in idx]
                fs = [fs[i] for i in idx]
                
                # Centroid of best d points
                centroid = np.mean(simplex[:d], axis=0)
                
                # Reflect
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = func(np.clip(xr, -5, 5))
                calls += 1
                if fr < best_f:
                    best_f, best_x = fr, xr.copy()
                    
                if fr < fs[0]:
                    # Expand
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = func(np.clip(xe, -5, 5))
                    calls += 1
                    if fe < best_f:
                        best_f, best_x = fe, xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        fs[-1] = fe
                    else:
                        simplex[-1] = xr
                        fs[-1] = fr
                elif fr < fs[-2]:
                    simplex[-1] = xr
                    fs[-1] = fr
                else:
                    # Contract
                    if fr < fs[-1]:
                        # Outside contraction
                        xc = centroid + sigma_ * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - sigma_ * (xr - centroid)
                        
                    fc = func(np.clip(xc, -5, 5))
                    calls += 1
                    if fc < best_f:
                        best_f, best_x = fc, xc.copy()
                        
                    if fc < min(fr, fs[-1]):
                        simplex[-1] = xc
                        fs[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fs[i] = func(np.clip(simplex[i], -5, 5))
                            calls += 1
                            if fs[i] < best_f:
                                best_f, best_x = fs[i], simplex[i].copy()
                            if check_deadline():
                                return

        # Phase 1: Grid
        num_starts = 20
        budget_per_start = 20
        
        starts = [np.zeros(d)]
        rng = np.random.default_rng(0)
        for _ in range(num_starts - 1):
            starts.append(rng.uniform(-4, 4, d))
            
        for x0 in starts:
            if check_deadline():
                break
            calls_start = calls
            nelder_mead(x0, budget_per_start)
            
        # Phase 2: Polish
        if best_x is not None and not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            remaining = budget - calls
            
            for _ in range(remaining):
                if check_deadline():
                    break
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                    
        return best_x, best_f
