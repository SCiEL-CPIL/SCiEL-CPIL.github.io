import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        rng = np.random.default_rng(12345)
        dim = self.dim
        
        # Initial point: Random in the box
        x0 = rng.uniform(-4, 4, dim)
        
        # Initial simplex size
        delta = 2.0
        
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        for i in range(dim):
            simplex[i+1] = x0.copy()
            simplex[i+1, i] += delta
        
        # Clip to bounds
        simplex = np.clip(simplex, -5, 5)
        
        # Evaluate initial simplex
        fvals = np.array([func(x) for x in simplex])
        calls = dim + 1
        
        # Sort simplex by function value
        idx = np.argsort(fvals)
        simplex = simplex[idx]
        fvals = fvals[idx]
        
        best_x = simplex[0].copy()
        best_f = fvals[0]
        
        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrinkage
        
        # Max iterations to stay within budget
        # Each iteration uses at most dim+1 calls (worst case: shrinkage)
        # Typically 1-2 calls.
        # Budget is 500. Dim=5.
        # 500 / 6 = 83.
        max_iter = 80
        
        for _ in range(max_iter):
            if calls >= self.budget:
                break
            if time.time() > deadline:
                break
                
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                # Accept reflection
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (xr - centroid)
                    
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                
                if fc < min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrinkage
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if calls >= self.budget:
                            break
                    if calls >= self.budget:
                        break
            
            # Re-sort simplex
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()
                
        return best_x, best_f
