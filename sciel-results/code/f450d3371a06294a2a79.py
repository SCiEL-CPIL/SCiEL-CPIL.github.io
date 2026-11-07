"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sequential global-to-local optimizer: (1+1)-ES for exploration, Nelder-Mead for polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _nelder_mead(self, func, x0, f0, dim, max_calls, deadline, rng):
        """Nelder-Mead simplex method for local refinement."""
        # Initialize simplex
        simplex = [x0.copy()]
        fx_list = [f0]
        
        # Initial step size for simplex vertices
        step = 0.5
        
        # Create remaining vertices
        for i in range(dim):
            vi = x0.copy()
            # Choose direction to stay within bounds
            if vi[i] < 5.0 - step:
                vi[i] += step
            else:
                vi[i] -= step
            vi = np.clip(vi, -5, 5)
            simplex.append(vi)
            fx_list.append(func(vi))
        
        calls = dim + 1
        
        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma_nm = 0.5 # shrink
        
        # Main loop
        while calls < max_calls:
            if time.time() >= deadline:
                break
            
            # Sort simplex by function value
            order = np.argsort(fx_list)
            simplex = [simplex[i].copy() for i in order]
            fx_list = [fx_list[i] for i in order]
            
            best_x = simplex[0]
            best_f = fx_list[0]
            worst_x = simplex[-1]
            worst_f = fx_list[-1]
            
            # Check convergence
            if abs(worst_f - best_f) < 1e-12:
                break
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - worst_x)
            xr = np.clip(xr, -5, 5)
            fxr = func(xr)
            calls += 1
            
            if fxr < fx_list[0]:
                # Expansion
                xe = centroid + gamma * (centroid - worst_x)
                xe = np.clip(xe, -5, 5)
                fxe = func(xe)
                calls += 1
                
                if fxe < fxr:
                    simplex[-1] = xe
                    fx_list[-1] = fxe
                else:
                    simplex[-1] = xr
                    fx_list[-1] = fxr
            elif fxr < fx_list[-2]:
                # Simple reflection
                simplex[-1] = xr
                fx_list[-1] = fxr
            else:
                # Contraction
                if fxr < worst_f:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - worst_x)
                
                xc = np.clip(xc, -5, 5)
                fxc = func(xc)
                calls += 1
                
                if fxc < min(fxr, worst_f):
                    simplex[-1] = xc
                    fx_list[-1] = fxc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fx_list[i] = func(simplex[i])
                        calls += 1
                        if calls >= max_calls:
                            break
                    if calls >= max_calls:
                        break
        
        # Return best point
        order = np.argsort(fx_list)
        return simplex[order[0]], fx_list[order[0]], calls

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        budget = self.budget
        dim = self.dim
        
        # Phase 1: Global exploration with (1+1)-ES
        # Reserve calls for Nelder-Mead polish
        nm_budget = max(50, budget // 5)  # ~100 calls for polish
        es_budget = budget - nm_budget
        
        # Multiple (1+1)-ES runs with different starting points
        n_runs = 3
        calls_per_run = es_budget // n_runs
        
        best_x = None
        best_f = np.inf
        total_calls = 0
        
        # Diverse initial sigmas
        initial_sigmas = [3.0, 2.0, 1.5]
        
        for run_idx in range(n_runs):
            if time.time() >= deadline - 30:
                break
            if total_calls >= es_budget:
                break
            
            # Each run starts from a different random point
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            total_calls += 1
            sigma = initial_sigmas[run_idx]
            
            # (1+1)-ES with 1/5 success rule
            for _ in range(calls_per_run - 1):
                if time.time() >= deadline - 30:
                    break
                if total_calls >= es_budget:
                    break
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                else:
                    sigma *= 0.87
                
                # Keep sigma within reasonable bounds
                sigma = float(np.clip(sigma, 0.005, 10.0))
            
            # Update global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Local refinement with Nelder-Mead
        if best_x is not None and total_calls < budget - 10 and time.time() < deadline - 20:
            remaining = budget - total_calls - 5
            nm_calls = min(nm_budget, remaining)
            
            x_nm, f_nm, nm_used = self._nelder_mead(
                func, best_x, best_f, dim, nm_calls, deadline - 10, rng
            )
            total_calls += nm_used
            
            if f_nm < best_f:
                best_f = f_nm
                best_x = x_nm.copy()
        
        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            best_f = func(best_x)
            total_calls += 1
        
        return best_x, best_f
# EVOLVE-BLOCK-END
