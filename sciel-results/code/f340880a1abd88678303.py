import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        calls = [0]
        best_val = [np.inf]
        best_x = [np.zeros(n)]

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            calls[0] += 1
            val = func(x)
            if val < best_val[0]:
                best_val[0] = val
                best_x[0] = x.copy()
            return val

        def nelder_mead(x0, max_iter=50, tol=1e-10):
            """
            Simple Nelder-Mead implementation with call counting.
            """
            n = len(x0)
            simplex = [x0.copy() for _ in range(n + 1)]
            # Initialize simplex
            for i in range(1, n + 1):
                simplex[i][i - 1] += 0.05 * max(1.0, abs(x0[i - 1]))

            # Evaluate initial simplex
            for i in range(n + 1):
                simplex[i] = np.clip(simplex[i], -5, 5)
                f_vals = [wrapped_func(simplex[i])]

            for _ in range(max_iter):
                if time.time() > deadline:
                    break

                # Sort simplex by function values
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]

                if abs(f_vals[-1] - f_vals[0]) < tol:
                    break

                # Centroid of best n points
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + 1.0 * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = wrapped_func(xr)

                if f_vals[0] <= fr < f_vals[-2]:
                    simplex[-1] = xr
                    f_vals[-1] = fr
                elif fr < f_vals[0]:
                    # Expansion
                    xe = centroid + 2.0 * (xr - centroid)
                    xe = np.clip(xe, -5, 5)
                    fe = wrapped_func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        f_vals[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_vals[-1] = fr
                else:
                    # Contraction
                    if fr < f_vals[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = wrapped_func(xc)

                    if fc < min(fr, f_vals[-1]):
                        simplex[-1] = xc
                        f_vals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_vals[i] = wrapped_func(simplex[i])

            # Return best point
            order = np.argsort(f_vals)
            return simplex[order[0]], f_vals[order[0]]

        # Strategy:
        # 1. Start with a few random initial points to get a sense of the landscape.
        # 2. Use Nelder-Mead to polish the best point found.
        # 3. If time/calls remain, try a few more random restarts and polish them.
        
        # Phase 1: Random initialization
        n_init = 5
        init_points = []
        for _ in range(n_init):
            if calls[0] >= budget:
                break
            x = np.random.uniform(-5, 5, n)
            init_points.append(x)
            
        for x in init_points:
            if calls[0] >= budget:
                break
            f = wrapped_func(x)

        # Phase 2: Polish the best point with Nelder-Mead
        if calls[0] < budget:
            # Reserve some calls for potential restarts
            # Let's use most of the budget for the first polish
            # Nelder-Mead in 5D takes about 10-20 iterations to converge, 
            # each iteration ~ 1-5 calls. So ~50-100 calls per polish.
            # We have 500 calls. Let's do 1 major polish, then use remaining for restarts.
            
            # Determine max iterations for NM based on remaining budget
            # Rough estimate: 5 calls per iteration
            max_iter_nm = max(10, (budget - calls[0]) // 10)
            
            x_best, f_best = nelder_mead(best_x[0], max_iter=max_iter_nm)
            
            # Phase 3: If we still have significant budget, do a few more random restarts + polish
            # But be careful not to exceed budget.
            # Let's say if we have > 50 calls left, try 1 more restart.
            while calls[0] < budget - 20 and time.time() < deadline:
                # Random restart
                x_new = np.random.uniform(-5, 5, n)
                f_new = wrapped_func(x_new)
                
                if f_new < best_val[0]:
                    best_x[0] = x_new
                    best_val[0] = f_new
                    
                # Polish this new point with a smaller NM
                max_iter_small = max(5, (budget - calls[0]) // 20)
                x_polished, f_polished = nelder_mead(x_new, max_iter=max_iter_small)
                
                if f_polished < best_val[0]:
                    best_x[0] = x_polished
                    best_val[0] = f_polished

        return best_x[0], best_val[0]
