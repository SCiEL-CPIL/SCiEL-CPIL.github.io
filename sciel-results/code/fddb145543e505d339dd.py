import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        calls = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def safe_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            calls += 1
            try:
                f_val = float(func(x))
            except Exception:
                f_val = float('inf')
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Smart initialization - try several random points to find a good starting region
        num_initial = 10
        candidates = []
        for _ in range(num_initial):
            if time.time() > deadline or calls >= budget:
                break
            x_init = rng.uniform(-4.5, 4.5, dim)
            f_init = safe_func(x_init)
            candidates.append((x_init, f_init))
        
        if not candidates:
            return best_x, best_f
        
        # Pick the best initial point
        candidates.sort(key=lambda x: x[1])
        x0, f0 = candidates[0]
        
        # Phase 2: Powell's Method
        # Powell is excellent for functions with narrow valleys (like Bent Cigar) because
        # it aligns its search directions with the valley.
        # Nelder-Mead can struggle with very narrow valleys if the simplex gets skewed.
        
        def powell_func(x):
            return safe_func(x)
            
        try:
            res = minimize(
                powell_func, 
                x0, 
                method='Powell',
                bounds=[(-5, 5)] * dim,
                options={
                    'maxiter': 1000,
                    'xatol': 1e-8,
                    'fatol': 1e-12,
                    'disp': False
                }
            )
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
                
        except Exception:
            pass
            
        # Phase 3: Refine with Nelder-Mead if budget remains
        # Powell should be very effective. If we have time/calls left, we can do a final polish.
        
        if calls < budget and time.time() < deadline - 5:
             # Try a quick Nelder-Mead refinement from the Powell result
             x_nm = best_x.copy()
             f_nm = best_f
             
             # Initialize simplex around x_nm with small delta
             delta = 0.1
             simplex = [x_nm.copy()]
             simplex_f = [f_nm]
             
             for i in range(dim):
                 x_i = x_nm.copy()
                 x_i[i] += delta
                 x_i = np.clip(x_i, -5.0, 5.0)
                 f_i = safe_func(x_i)
                 simplex.append(x_i)
                 simplex_f.append(f_i)
                 
                 if time.time() > deadline or calls >= budget:
                     break
             
             # Simple NM loop for refinement
             rho = 1.0
             chi = 2.0
             gamma = 0.5
             sigma = 0.5
             
             for _ in range(50): # Small cap for refinement
                 if time.time() > deadline or calls >= budget:
                     break
                 
                 indices = np.argsort(simplex_f)
                 simplex = [simplex[i] for i in indices]
                 simplex_f = [simplex_f[i] for i in indices]
                 
                 if simplex_f[-1] - simplex_f[0] < 1e-10:
                     break
                 
                 centroid = np.mean(simplex[:-1], axis=0)
                 
                 xr = centroid + rho * (centroid - simplex[-1])
                 xr = np.clip(xr, -5.0, 5.0)
                 fr = safe_func(xr)
                 
                 if fr < simplex_f[0]:
                     xe = centroid + chi * (xr - centroid)
                     xe = np.clip(xe, -5.0, 5.0)
                     fe = safe_func(xe)
                     if fe < fr:
                         simplex[-1] = xe
                         simplex_f[-1] = fe
                     else:
                         simplex[-1] = xr
                         simplex_f[-1] = fr
                 elif fr < simplex_f[-2]:
                     simplex[-1] = xr
                     simplex_f[-1] = fr
                 else:
                     xc = centroid + gamma * (simplex[-1] - centroid)
                     xc = np.clip(xc, -5.0, 5.0)
                     fc = safe_func(xc)
                     if fc < simplex_f[-1]:
                         simplex[-1] = xc
                         simplex_f[-1] = fc
                     else:
                         for i in range(1, dim + 1):
                             simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                             simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                             simplex_f[i] = safe_func(simplex[i])
                         
                         indices = np.argsort(simplex_f)
                         simplex = [simplex[i] for i in indices]
                         simplex_f = [simplex_f[i] for i in indices]
                         
                         if simplex_f[0] < best_f:
                             best_f = simplex_f[0]
                             best_x = simplex[0].copy()

        return best_x, best_f
