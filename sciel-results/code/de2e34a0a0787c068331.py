"""Optimized search for BBOB f13 Sharp Ridge with coordinate descent."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer combining:
    1. Random sampling for global exploration
    2. Coordinate descent with adaptive step sizes for ridge following
    3. Nelder-Mead for local refinement
    4. Ridge-following via directional exploration
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # Safety margin before 300s limit
        
        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)
        
        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                f = func(x)
            except:
                return None
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Global exploration with random sampling
        n_random = max(20, min(60, budget // 10))
        
        for i in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            safe_call(x)
        
        if best_x is None:
            x = np.zeros(dim)
            safe_call(x)
            return best_x, best_f

        # Phase 2: Coordinate Descent with adaptive step sizes
        # This is key for sharp ridge - follow the ridge by optimizing one coordinate at a time
        center = best_x.copy()
        steps = np.ones(dim) * 0.5  # Initial step sizes per dimension
        
        # Adaptive coordinate descent
        cd_iters = 0
        max_cd_iters = 200
        stagnant_cd = 0
        best_f_cd = best_f
        
        while calls < budget and cd_iters < max_cd_iters:
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Try positive and negative steps along dimension i
                for sign in [1, -1]:
                    x_try = center.copy()
                    x_try[i] += sign * steps[i]
                    x_try = np.clip(x_try, -5, 5)
                    f_try = safe_call(x_try)
                    if f_try is None:
                        break
                    
                    if f_try < best_f - 1e-10:
                        # Accept and move
                        center = x_try.copy()
                        best_f = f_try
                        best_x = x_try.copy()
                        improved = True
                        # Increase step if we improved
                        steps[i] *= 1.3
                    else:
                        # Shrink step
                        steps[i] *= 0.7
                
                if calls >= budget or time.time() > deadline:
                    break
            
            cd_iters += 1
            
            if not improved:
                stagnant_cd += 1
                if stagnant_cd > 5:
                    break
            else:
                stagnant_cd = 0
            
            # Check convergence
            if np.max(steps) < 1e-6:
                break
        
        # Phase 3: Nelder-Mead for local refinement
        if calls < budget and time.time() < deadline:
            center = best_x.copy()
            simplex = [center.copy()]
            f_values = [best_f]
            
            step = 0.3
            for i in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                xi = center.copy()
                xi[i] += step
                fi = safe_call(xi)
                if fi is None:
                    break
                simplex.append(xi)
                f_values.append(fi)
            
            if calls < budget and len(simplex) >= dim + 1:
                simplex = np.array(simplex)
                f_values = np.array(f_values)
                
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma = 0.5
                
                nm_iters = 0
                max_nm_iters = 300
                stagnant_nm = 0
                best_f_nm = best_f
                
                while calls < budget and nm_iters < max_nm_iters:
                    if time.time() > deadline:
                        break
                    
                    idx = np.argsort(f_values)
                    simplex = simplex[idx]
                    f_values = f_values[idx]
                    
                    spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
                    f_spread = np.max(f_values) - np.min(f_values)
                    
                    if np.max(spread) < 1e-8 or f_spread < 1e-12:
                        break
                    
                    if f_values[0] >= best_f_nm - 1e-10:
                        stagnant_nm += 1
                    else:
                        best_f_nm = f_values[0]
                        stagnant_nm = 0
                    
                    if stagnant_nm > 15:
                        break
                    
                    nm_iters += 1
                    
                    centroid = np.mean(simplex[:-1], axis=0)
                    xr = centroid + alpha * (centroid - simplex[-1])
                    xr = np.clip(xr, -5, 5)
                    fr = safe_call(xr)
                    if fr is None:
                        break
                    
                    if fr < f_values[0]:
                        xe = centroid + gamma * (centroid - simplex[-1])
                        xe = np.clip(xe, -5, 5)
                        fe = safe_call(xe)
                        if fe is None:
                            break
                        if fe < fr:
                            simplex[-1] = xe
                            f_values[-1] = fe
                        else:
                            simplex[-1] = xr
                            f_values[-1] = fr
                    elif fr < f_values[-2]:
                        simplex[-1] = xr
                        f_values[-1] = fr
                    else:
                        if fr < f_values[-1]:
                            xc = centroid + rho * (xr - centroid)
                        else:
                            xc = centroid - rho * (centroid - simplex[-1])
                        xc = np.clip(xc, -5, 5)
                        fc = safe_call(xc)
                        if fc is None:
                            break
                        if fc <= min(fr, f_values[-1]):
                            simplex[-1] = xc
                            f_values[-1] = fc
                        else:
                            for i in range(1, len(simplex)):
                                if calls >= budget or time.time() > deadline:
                                    break
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fi = safe_call(simplex[i])
                                if fi is None:
                                    break
                                f_values[i] = fi
                            if fi is None:
                                break
        
        # Phase 4: Final local search
        if calls < budget and time.time() < deadline:
            n_final = min(40, budget - calls)
            for _ in range(n_final):
                if calls >= budget or time.time() > deadline:
                    break
                scale = max(0.05, 0.3 * np.exp(-calls / 150.0))
                x_try = best_x + rng.normal(0, scale, dim)
                x_try = np.clip(x_try, -5, 5)
                safe_call(x_try)
            
            for i in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                for delta in [-0.2, -0.05, 0.05, 0.2]:
                    if calls >= budget or time.time() > deadline:
                        break
                    x_try = best_x.copy()
                    x_try[i] += delta
                    x_try = np.clip(x_try, -5, 5)
                    safe_call(x_try)
        
        return best_x, best_f

# EVOLVE-BLOCK-END
