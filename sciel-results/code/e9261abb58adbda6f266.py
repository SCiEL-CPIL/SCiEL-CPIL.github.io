import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill

        def _time_ok():
            return time.time() - start_time < time_limit

        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls = 0

        # Helper to safely evaluate a function, handling exceptions
        def safe_eval(x):
            nonlocal calls
            if calls >= budget or not _time_ok():
                return np.inf
            try:
                fx = func(x)
                calls += 1
                if not np.isfinite(fx):
                    return np.inf
                return float(fx)
            except Exception:
                calls += 1
                return np.inf

        # Phase 1: Initial exploration
        # We need to find a good starting point for Nelder-Mead.
        # For multimodal functions (Rastrigin, Gallagher), random sampling is key.
        # For unimodal (Sphere, Rosenbrock), random sampling in the box is also fine, 
        # but Nelder-Mead will quickly converge.
        
        n_init = 20
        starts = []
        
        for i in range(n_init):
            if not _time_ok() or calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f = safe_eval(x)
            starts.append((x.copy(), f))
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if best_x is None:
            # Fallback if something went wrong
            return np.zeros(dim), 0.0

        # Sort starts by fitness
        starts.sort(key=lambda t: t[1])
        
        # Phase 2: Nelder-Mead Simplex Optimization
        # We will run Nelder-Mead from the best starting point.
        # We might also run it from the second best if budget allows and 
        # the first one seems stuck (though NM is robust).
        # Given the budget (300), 20 calls for init leaves 280 for NM.
        # NM typically uses dim+1 points for the simplex.
        # Each iteration uses 1-2 calls (usually 1 for reflection, sometimes more).
        # 280 calls is plenty for 5D NM to converge to high precision.

        def nelder_mead(x0, f0, max_iter=200):
            nonlocal calls, best_x, best_f
            
            # Initialize simplex
            # x0 is the best point.
            # Create other points by perturbing x0
            alpha = 0.25
            beta = 0.5
            gamma = 2.0
            delta = 0.5
            
            simplex = [x0.copy()]
            simplex_f = [f0]
            
            for i in range(dim):
                xi = x0.copy()
                # Perturb in direction i
                if abs(xi[i]) > 1e-8:
                    xi[i] *= (1 + alpha)
                else:
                    xi[i] += alpha * beta
                # Ensure within bounds
                xi = np.clip(xi, -5, 5)
                fi = safe_eval(xi)
                simplex.append(xi)
                simplex_f.append(fi)
                
                # Update global best
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            # Sort simplex by fitness
            # We maintain a list of (f, x)
            def sort_simplex():
                # Sort indices by f
                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]
                
            sort_simplex()
            
            for _ in range(max_iter):
                if not _time_ok() or calls >= budget:
                    break
                
                # Check convergence: if best and worst are very close, stop
                if len(simplex_f) > 0:
                    f_min = simplex_f[0]
                    f_max = simplex_f[-1]
                    if abs(f_max - f_min) < 1e-8 * (1 + abs(f_min)):
                        break
                
                # 1. Reflection
                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + gamma * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = safe_eval(xr)
                
                if fr < simplex_f[0]:
                    # Try expansion
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = safe_eval(xe)
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
                    # Contraction
                    # Outside contraction
                    xc_out = centroid - 0.5 * (centroid - simplex[-1])
                    xc_out = np.clip(xc_out, -5, 5)
                    fc_out = safe_eval(xc_out)
                    
                    if fc_out < simplex_f[-1]:
                        simplex[-1] = xc_out
                        simplex_f[-1] = fc_out
                    else:
                        # Inside contraction
                        xc_in = centroid + 0.5 * (centroid - simplex[-1])
                        xc_in = np.clip(xc_in, -5, 5)
                        fc_in = safe_eval(xc_in)
                        
                        if fc_in < simplex_f[-1]:
                            simplex[-1] = xc_in
                            simplex_f[-1] = fc_in
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                simplex_f[i] = safe_eval(simplex[i])
                
                # Update global best
                for i in range(dim + 1):
                    if simplex_f[i] < best_f:
                        best_f = simplex_f[i]
                        best_x = simplex[i].copy()
                
                sort_simplex()

        # Run Nelder-Mead from the best start
        x_start, f_start = starts[0]
        nelder_mead(x_start, f_start)
        
        # If we have budget left and the function is likely multimodal,
        # we could try a second NM run from the second best start.
        # However, NM is very efficient. Let's check if we have significant budget left.
        if calls < budget - 10 and _time_ok() and len(starts) > 1:
            # Only do second run if the first one didn't use too much budget 
            # and the second start is significantly better than the initial worst?
            # Actually, just run it if we have > 50 calls left.
            if budget - calls > 50:
                x_start2, f_start2 = starts[1]
                # To avoid redundant work, only if it's not the same point
                if not np.allclose(x_start, x_start2):
                    nelder_mead(x_start2, f_start2)

        return best_x, best_f
