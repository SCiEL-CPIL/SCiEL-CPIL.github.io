import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin
        d = self.dim
        budget = self.budget
        
        n_calls = 0
        best_x = np.zeros(d)
        best_f = np.inf
        rng = np.random.default_rng(42)

        def check_time():
            return time.time() < deadline and n_calls < budget

        def evaluate(x):
            nonlocal n_calls, best_x, best_f
            if not check_time():
                return None
            x = np.asarray(x, dtype=float)
            if x.shape != (d,) or not np.all(np.isfinite(x)):
                return None
            try:
                fx = func(x)
            except Exception:
                return None
            n_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Exploration
        # Sample points in [-4, 4]^d to cover the likely region of the optimum
        n_global = min(60, budget // 5)
        for _ in range(n_global):
            if not check_time():
                break
            x = rng.uniform(-4, 4, d)
            evaluate(x)
            
        # Ensure we have a valid best_x
        if n_calls == 0:
            evaluate(np.zeros(d))
            
        # Phase 2: Nelder-Mead Simplex Method
        # Initialize simplex
        x0 = best_x.copy()
        simplex = [x0.copy()]
        # Add other points with small perturbations
        initial_step = 0.5
        for i in range(d):
            xi = x0.copy()
            xi[i] += initial_step
            simplex.append(xi)
        
        # Evaluate initial simplex
        simplex_f = []
        for sx in simplex:
            fx = evaluate(sx)
            if fx is None:
                break
            simplex_f.append(fx)
        
        # If we ran out of budget during initialization, return best
        if len(simplex_f) < d + 1:
            return best_x, best_f

        # Nelder-Mead Parameters
        alpha = 1.0    # Reflection
        gamma = 2.0    # Expansion
        rho = 0.5      # Contraction
        sigma = 0.5    # Shrinkage
        max_iter = 200 # Max iterations for NM
        
        # Main NM Loop
        for _ in range(max_iter):
            if not check_time():
                break
                
            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fxr = evaluate(xr)
            if fxr is None:
                break
            
            if simplex_f[0] <= fxr < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = xr
                simplex_f[-1] = fxr
            elif fxr < simplex_f[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fxe = evaluate(xe)
                if fxe is None:
                    break
                if fxe < fxr:
                    simplex[-1] = xe
                    simplex_f[-1] = fxe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
            else:
                # Contraction
                if fxr < simplex_f[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                fxc = evaluate(xc)
                if fxc is None:
                    break
                if fxc <= min(fxr, simplex_f[-1]):
                    simplex[-1] = xc
                    simplex_f[-1] = fxc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fxi = evaluate(simplex[i])
                        if fxi is None:
                            simplex_f[i] = np.inf
                        else:
                            simplex_f[i] = fxi
                    if not check_time():
                        break

            # Check for convergence: if simplex is very small, break
            spread = np.max(np.abs(np.array(simplex) - np.mean(simplex, axis=0)))
            if spread < 1e-4:
                break

        # Phase 3: 1D Line Search Polishing (Golden Section Search)
        # This helps to slide along flat regions and fine-tune
        if check_time():
            # We have some budget left. Let's use it for 1D optimization.
            # We will do a few rounds of 1D optimization along each axis.
            n_polish_rounds = 3
            for _ in range(n_polish_rounds):
                if not check_time():
                    break
                for i in range(d):
                    if not check_time():
                        break
                    # Define 1D function along axis i
                    def func_1d(t):
                        x = best_x.copy()
                        x[i] = t
                        x = np.clip(x, -5, 5)
                        try:
                            return func(x)
                        except Exception:
                            return np.inf
                    
                    # Golden Section Search
                    # We need an interval [a, b] containing the minimum.
                    # Let's use a window around the current best.
                    # Since we don't know the width, we start with a small window and expand if needed.
                    # However, expanding costs calls. Let's just use a fixed reasonable window based on previous step size or a default.
                    # A safe default is [-1, 1] around the current coordinate, clipped to [-5, 5].
                    a = max(-5, best_x[i] - 1.0)
                    b = min(5, best_x[i] + 1.0)
                    
                    if b - a < 1e-5:
                        continue

                    # Golden ratio
                    gr = (np.sqrt(5) - 1) / 2
                    c = b - gr * (b - a)
                    d_val = a + gr * (b - a)
                    
                    fc = func_1d(c)
                    fd = func_1d(d_val)
                    
                    # Limit number of evaluations per 1D search to save budget
                    max_1d_evals = 10
                    evals_1d = 2
                    
                    for _ in range(max_1d_evals):
                        if not check_time():
                            break
                        if fc < fd:
                            b = d_val
                            d_val = c
                            fd = fc
                            c = b - gr * (b - a)
                            fc = func_1d(c)
                        else:
                            a = c
                            c = d_val
                            fc = fd
                            d_val = a + gr * (b - a)
                            fd = func_1d(d_val)
                        evals_1d += 1
                        if b - a < 1e-4:
                            break
                    
                    # Update best_x with the best point found in 1D search
                    # The best point is either c or d_val, or the midpoint of the final interval
                    # Actually, the minimum is in [a, b]. Let's evaluate the midpoint.
                    mid = (a + b) / 2
                    fm = func_1d(mid)
                    
                    # Compare mid with current best_x
                    # Note: func_1d evaluates func, so we must be careful not to double count if we just want to update x.
                    # But we need the function value to compare.
                    # Since func_1d calls func, n_calls is incremented.
                    
                    # If fm is better than the current best_f, update.
                    # However, best_f is updated inside evaluate(). Here we are calling func directly.
                    # We should use evaluate() for consistency, but func_1d is a wrapper.
                    # Let's just update best_x if the 1D search found a better value.
                    # We need to track the best value found in this 1D search.
                    # The golden section search maintains the best in the shrinking interval.
                    # The best point is likely near the final midpoint.
                    
                    # To be safe and consistent with the global best tracking:
                    # We will evaluate the final candidate point using evaluate().
                    # But we already evaluated mid.
                    # Let's just update best_x to the point that gave the min value among c, d_val, mid.
                    # We have fc, fd, fm.
                    min_val = min(fc, fd, fm)
                    if min_val == fc:
                        new_x = best_x.copy()
                        new_x[i] = c
                    elif min_val == fd:
                        new_x = best_x.copy()
                        new_x[i] = d_val
                    else:
                        new_x = best_x.copy()
                        new_x[i] = mid
                    
                    # Check if this new_x is better than global best
                    # We don't want to waste a call if it's not.
                    # But we don't know the value of new_x without calling.
                    # However, we already called func for c, d_val, mid.
                    # So we know the values.
                    if min_val < best_f:
                        best_x = new_x
                        best_f = min_val

        return best_x, best_f
