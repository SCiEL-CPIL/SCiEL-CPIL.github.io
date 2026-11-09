"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nm_polish(self, func, x0, budget_left, deadline, best_state):
        """Nelder-Mead local search."""
        d = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1
        
        # Update global best
        if fx < best_state['best_f']:
            best_state['best_f'] = fx
            best_state['best_x'] = x.copy()
        
        # Initial simplex
        simplex = [x.copy()]
        simplex_f = [fx]
        
        delta = 0.5
        for i in range(d):
            if calls >= budget_left or time.time() > deadline:
                break
            xi = x.copy()
            xi[i] += delta
            xi = np.clip(xi, -5, 5)
            simplex.append(xi)
            simplex_f.append(func(xi))
            calls += 1
            
            if simplex_f[-1] < best_state['best_f']:
                best_state['best_f'] = simplex_f[-1]
                best_state['best_x'] = xi.copy()
        
        if len(simplex) < d + 1:
            return best_state['best_x'], best_state['best_f']

        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        max_iter = 100
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
            if fr < best_state['best_f']:
                best_state['best_f'] = fr
                best_state['best_x'] = xr.copy()
                
            if calls >= budget_left or time.time() > deadline:
                if fr < simplex_f[-1]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                break
                
            if simplex_f[0] <= fr < simplex_f[-2]:
                simplex[-1] = xr
                simplex_f[-1] = fr
            elif fr < simplex_f[0]:
                if calls >= budget_left or time.time() > deadline:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                    break
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls += 1
                if fe < best_state['best_f']:
                    best_state['best_f'] = fe
                    best_state['best_x'] = xe.copy()
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
                if calls >= budget_left or time.time() > deadline:
                    break
                if fr < simplex_f[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls += 1
                if fc < best_state['best_f']:
                    best_state['best_f'] = fc
                    best_state['best_x'] = xc.copy()
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
                        if calls >= budget_left or time.time() > deadline:
                            break
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = func(simplex[i])
                        calls += 1
                        if simplex_f[i] < best_state['best_f']:
                            best_state['best_f'] = simplex_f[i]
                            best_state['best_x'] = simplex[i].copy()
                        
            min_idx = np.argmin(simplex_f)
            if simplex_f[min_idx] < best_f:
                best_f = simplex_f[min_idx]
                best_x = simplex[min_idx].copy()

        return best_state['best_x'], best_state['best_f']

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0
        
        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Coarse Halton search (60% of budget)
        # Halton sequence gives good space-filling properties
        n_halton = int(budget * 0.6)
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r
        
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        
        # Generate Halton points in [0,1]^d then map to [-5,5]^d
        for i in range(1, n_halton + 1):
            if time.time() > deadline:
                break
            if calls >= budget - 50:  # Reserve 50 for local refinement
                break
            try:
                x = np.array([halton(i, p) for p in primes[:dim]])
                # Map [0,1] to [-5,5]
                x = x * 10.0 - 5.0
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls += 1
                update_best(x, fx)
            except Exception:
                continue

        # Phase 2: Coarse-to-fine lattice refinement around best point
        # Take the best point and do a local grid search with decreasing step size
        if best_f < np.inf and calls < budget - 10:
            # Local refinement: try several step sizes
            step_sizes = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]
            for step in step_sizes:
                if calls >= budget - 5:
                    break
                if time.time() > deadline:
                    break
                
                # Generate neighbors on a grid around best_x
                # In 5D, a full grid is expensive, so use coordinate-wise steps
                for i in range(dim):
                    if calls >= budget - 5:
                        break
                    for sign in [-1, 1]:
                        if calls >= budget - 5:
                            break
                        x_new = np.array(best_x, dtype=float)
                        x_new[i] += sign * step
                        x_new = np.clip(x_new, -5, 5)
                        try:
                            fx = func(x_new)
                            calls += 1
                            update_best(x_new, fx)
                        except Exception:
                            break

        # Phase 3: Nelder-Mead local refinement on best point
        if best_f < np.inf and calls < budget - 5:
            remaining = budget - calls
            if remaining >= 10:
                best_state = {'best_x': best_x, 'best_f': best_f}
                try:
                    x_polished, f_polished = self._nm_polish(
                        func, best_x, remaining, deadline, best_state
                    )
                    best_x = best_state['best_x']
                    best_f = best_state['best_f']
                except Exception:
                    pass

        # Ensure best_x is valid
        if best_f == np.inf:
            try:
                best_x = np.zeros(dim)
                best_f = func(best_x)
                calls += 1
            except Exception:
                best_f = 0.0

        return best_x, best_f
