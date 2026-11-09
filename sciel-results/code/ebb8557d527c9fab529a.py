"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nm_polish(self, func, x0, budget_left, deadline):
        """Nelder-Mead local search."""
        d = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1
        
        # Initial simplex
        simplex = [x.copy()]
        simplex_f = [fx]
        
        delta = 0.1 # Smaller initial step for finer search
        for i in range(d):
            xi = x.copy()
            xi[i] += delta
            xi = np.clip(xi, -5, 5)
            simplex.append(xi)
            simplex_f.append(func(xi))
            calls += 1
            if calls >= budget_left or time.time() > deadline:
                break
        
        if len(simplex) < d + 1:
             return best_x, best_f

        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        max_iter = 50
        for _ in range(max_iter):
            if calls >= budget_left or time.time() > deadline:
                break
            
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            centroid = np.mean(simplex[:-1], axis=0)
            
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls += 1
            if calls >= budget_left or time.time() > deadline:
                if fr < simplex_f[-1]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                break
                
            if simplex_f[0] <= fr < simplex_f[-2]:
                simplex[-1] = xr
                simplex_f[-1] = fr
            elif fr < simplex_f[0]:
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                if calls >= budget_left or time.time() > deadline:
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                    break
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
            else:
                if fr < simplex_f[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                if calls >= budget_left or time.time() > deadline:
                    if fc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    break
                if fc < min(fr, simplex_f[-1]):
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                else:
                    for i in range(1, d + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget_left or time.time() > deadline:
                            break
                    if calls >= budget_left or time.time() > deadline:
                        break
                        
            min_idx = np.argmin(simplex_f)
            if simplex_f[min_idx] < best_f:
                best_f = simplex_f[min_idx]
                best_x = simplex[min_idx].copy()

        return best_x, best_f

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0
        
        calls = 0
        best_x = None
        best_f = np.inf
        
        # Phase 1: Halton Sequence Global Search
        # 80% budget
        halton_budget = int(budget * 0.8)
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29] # Enough for dim up to 10
        
        # Generate Halton points
        # We can generate them on the fly or in a batch.
        # Batch is better for speed if using vectorized ops, but Halton is simple.
        
        for i in range(1, halton_budget + 1):
            if time.time() > deadline:
                break
            
            x = np.zeros(dim)
            for j in range(dim):
                # Halton index i+1 to avoid 0
                f = 1.0
                r = 0.0
                idx = i + 1
                base = bases[j]
                while idx > 0:
                    f /= base
                    r += f * (idx % base)
                    idx //= base
                x[j] = r * 10.0 - 5.0 # Map [0,1) to [-5,5)
            
            fx = func(x)
            calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        # Phase 2: Local Refinement
        if best_x is not None and calls < budget and time.time() < deadline:
            remaining = budget - calls
            if remaining > 10:
                x_polished, f_polished = self._nm_polish(
                    func, best_x, remaining, deadline
                )
                if f_polished < best_f:
                    best_f = f_polished
                    best_x = x_polished

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
