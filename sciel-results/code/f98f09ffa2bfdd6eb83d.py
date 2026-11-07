"""Black-box continuous optimizer for the rotated Rosenbrock (f9) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive simplex size and high-precision polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 330s limit

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        # Nelder-Mead parameters
        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma = 0.5    # shrink
        
        # Initial simplex size - larger to capture wide Rosenbrock valley
        init_sigma = 1.0

        def nelder_mead(x0, f0, tol=1e-8, max_iter=1500):
            nonlocal calls, best_x, best_f
            
            if calls >= budget:
                return x0, f0

            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [f0]
            
            for i in range(n):
                if calls >= budget or time.time() > deadline:
                    return x0, f0
                xi = x0.copy()
                xi[i] += init_sigma
                simplex.append(xi)
                fi = func(xi)
                calls += 1
                fvals.append(fi)
                
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            # Sort simplex by function value
            indices = np.argsort(fvals)
            simplex = [simplex[i] for i in indices]
            fvals = [fvals[i] for i in indices]
            
            for _ in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    break
                
                # Check convergence
                f_range = max(fvals) - min(fvals)
                if f_range < tol:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if calls >= budget:
                    break

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    
                    if calls >= budget:
                        break

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
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()

                    if calls >= budget:
                        break

                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = func(simplex[i])
                            calls += 1
                            fvals[i] = fi
                            if fi < best_f:
                                best_f = fi
                                best_x = simplex[i].copy()
                            
                            if calls >= budget:
                                break
                        
                        if calls >= budget:
                            break

                # Re-sort simplex
                indices = np.argsort(fvals)
                simplex = [simplex[i] for i in indices]
                fvals = [fvals[i] for i in indices]

            return simplex[0], fvals[0]

        def polish(x0, f0, tol=1e-12, max_iter=500):
            """High-precision polish using very tight tolerance and small initial step."""
            nonlocal calls, best_x, best_f
            
            if calls >= budget:
                return x0, f0
            
            # Use a smaller initial simplex for polishing
            polish_sigma = 0.01
            
            simplex = [x0.copy()]
            fvals = [f0]
            
            for i in range(n):
                if calls >= budget or time.time() > deadline:
                    return x0, f0
                xi = x0.copy()
                xi[i] += polish_sigma
                simplex.append(xi)
                fi = func(xi)
                calls += 1
                fvals.append(fi)
                
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            indices = np.argsort(fvals)
            simplex = [simplex[i] for i in indices]
            fvals = [fvals[i] for i in indices]
            
            for _ in range(max_iter):
                if time.time() > deadline or calls >= budget:
                    break
                
                f_range = max(fvals) - min(fvals)
                if f_range < tol:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()

                if calls >= budget:
                    break

                if fr < fvals[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    
                    if calls >= budget:
                        break

                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    if fr < fvals[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid - rho * (xr - centroid)
                    
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()

                    if calls >= budget:
                        break

                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = func(simplex[i])
                            calls += 1
                            fvals[i] = fi
                            if fi < best_f:
                                best_f = fi
                                best_x = simplex[i].copy()
                            
                            if calls >= budget:
                                break
                        
                        if calls >= budget:
                            break

                indices = np.argsort(fvals)
                simplex = [simplex[i] for i in indices]
                fvals = [fvals[i] for i in indices]

            return simplex[0], fvals[0]

        # Reserve 15% of budget for polishing
        polish_budget = int(budget * 0.15)
        search_budget = budget - polish_budget

        # First start at 0 - good heuristic for rotated Rosenbrock
        x_start = np.zeros(n)
        f_start = func(x_start)
        calls += 1
        best_x = x_start.copy()
        best_f = f_start

        if calls >= search_budget:
            # Go straight to polish
            polish(best_x, best_f)
            return best_x, best_f

        # Run Nelder-Mead from 0
        x_curr, f_curr = nelder_mead(x_start, f_start, tol=1e-8, max_iter=1000)
        
        num_restarts_done = 1
        max_restarts = 8
        
        # Adaptive restart: continue while we're making progress and have budget
        for restart_idx in range(max_restarts - 1):
            if calls >= search_budget or time.time() > deadline:
                break
            
            # Generate a new random starting point
            x_new = rng.uniform(-5, 5, n)
            f_new = func(x_new)
            calls += 1
            
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

            if calls >= search_budget:
                break

            # Run Nelder-Mead from this point with slightly tighter tolerance
            x_curr, f_curr = nelder_mead(x_new, f_new, tol=1e-9, max_iter=800)
            num_restarts_done += 1
            
            # If we've converged well, we can stop exploring
            if best_f < 1e-4:
                if restart_idx >= 3:
                    break

        # High-precision polish pass on the best solution found
        if calls < budget and time.time() < deadline:
            polish(best_x, best_f, tol=1e-12, max_iter=400)

        return best_x, best_f
# EVOLVE-BLOCK-END
