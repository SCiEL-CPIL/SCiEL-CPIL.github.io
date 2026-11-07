"""Optimized search for BBOB f13 Sharp Ridge."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hybrid optimizer combining:
    1. Random restarts to find the ridge
    2. Nelder-Mead for local refinement
    3. Ridge-following via directional exploration
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
        # Use 15% of budget for global search
        n_random = max(30, min(75, budget // 7))
        
        for i in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            safe_call(x)
        
        if best_x is None:
            x = np.zeros(dim)
            safe_call(x)
            return best_x, best_f

        # Phase 2: Local refinement with Nelder-Mead
        # Initialize simplex around best point
        center = best_x.copy()
        simplex = [center.copy()]
        f_values = [best_f]
        
        # Initial step size - adaptive based on dimension
        step = 0.8
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
        
        if calls >= budget or len(simplex) < dim + 1:
            return best_x, best_f
            
        simplex = np.array(simplex)
        f_values = np.array(f_values)
        
        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        # Main NM loop with restarts
        nm_iters = 0
        max_nm_iters = 500
        stagnant_count = 0
        best_f_nm = best_f
        
        while calls < budget and nm_iters < max_nm_iters:
            if time.time() > deadline:
                break
                
            # Sort simplex by function value
            idx = np.argsort(f_values)
            simplex = simplex[idx]
            f_values = f_values[idx]
            
            # Check convergence
            spread = np.max(simplex, axis=0) - np.min(simplex, axis=0)
            f_spread = np.max(f_values) - np.min(f_values)
            
            if np.max(spread) < 1e-8 or f_spread < 1e-12:
                break
            
            # Stagnation detection
            if f_values[0] >= best_f_nm - 1e-10:
                stagnant_count += 1
            else:
                best_f_nm = f_values[0]
                stagnant_count = 0
            
            # If stagnant for too long, restart from best point with new direction
            if stagnant_count > 20:
                # Try a few random perturbations from best point
                for _ in range(5):
                    if calls >= budget or time.time() > deadline:
                        break
                    direction = rng.normal(0, 1, dim)
                    direction /= (np.linalg.norm(direction) + 1e-10)
                    for dist in [0.5, 1.0, 2.0]:
                        if calls >= budget or time.time() > deadline:
                            break
                        x_try = best_x + dist * direction
                        x_try = np.clip(x_try, -5, 5)
                        safe_call(x_try)
                
                # Re-initialize simplex
                center = best_x.copy()
                simplex = [center.copy()]
                f_values = [best_f]
                step = 0.5
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
                
                if len(simplex) < dim + 1:
                    break
                    
                simplex = np.array(simplex)
                f_values = np.array(f_values)
                stagnant_count = 0
                nm_iters = 0
                continue
            
            nm_iters += 1
            
            # Standard Nelder-Mead steps
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = safe_call(xr)
            if fr is None:
                break
            
            if fr < f_values[0]:
                # Expansion
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
                # Reflection accepted
                simplex[-1] = xr
                f_values[-1] = fr
            else:
                # Contraction
                if fr < f_values[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                fc = safe_call(xc)
                if fc is None:
                    break
                if fc <= min(fr, f_values[-1]):
                    simplex[-1] = xc
                    f_values[-1] = fc
                else:
                    # Shrink
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
        
        # Phase 3: Final local search around best point
        # Try small perturbations in all directions
        if calls < budget and time.time() < deadline:
            n_final = min(50, budget - calls)
            for _ in range(n_final):
                if calls >= budget or time.time() > deadline:
                    break
                # Gaussian perturbation around best point
                scale = max(0.1, 0.5 * np.exp(-calls / 100.0))
                x_try = best_x + rng.normal(0, scale, dim)
                x_try = np.clip(x_try, -5, 5)
                safe_call(x_try)
            
            # Also try axis-aligned perturbations
            for i in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                for delta in [-0.3, -0.1, 0.1, 0.3]:
                    if calls >= budget or time.time() > deadline:
                        break
                    x_try = best_x.copy()
                    x_try[i] += delta
                    x_try = np.clip(x_try, -5, 5)
                    safe_call(x_try)
        
        return best_x, best_f

# EVOLVE-BLOCK-END
