"""Optimizer for BBOB-style Sphere f1 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s kill

        dim = self.dim
        budget = self.budget
        calls = 0

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = np.zeros(dim)
        best_f = np.inf

        # Phase 1: Newton-style direct minimizer estimation
        # Sphere is f(x) = (x-x*)^T A (x-x*) + f_opt
        # Gradient at 0 is -2 A x*. 
        # If we assume A ~ I, step is x* approx.
        
        center = np.zeros(dim)
        fc = func(center)
        calls += 1
        if fc < best_f:
            best_f = fc
            best_x = center.copy()
        
        # Estimate gradient at center using finite differences
        h = 0.1
        grad = np.zeros(dim)
        for i in range(dim):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            xp = center.copy()
            xp[i] += h
            xp = clip(xp)
            fplus = func(xp)
            calls += 1
            if fplus < best_f:
                best_f = fplus
                best_x = xp.copy()
            
            xm = center.copy()
            xm[i] -= h
            xm = clip(xm)
            fminus = func(xm)
            calls += 1
            if fminus < best_f:
                best_f = fminus
                best_x = xm.copy()
            
            grad[i] = (fplus - fminus) / (2 * h)
        
        # Take a Newton-like step
        x = center.copy()
        fx = fc
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-6 and calls < budget:
            # For sphere, if A=I, grad = 2(x-x*). Step = -grad/2 = x* - x.
            # So x_new = x - 0.5 * grad
            # However, if A is not I, this is an approximation.
            # We use a damped step to be safe.
            step_size = 0.5 
            x_new = clip(x - step_size * grad)
            fx_new = func(x_new)
            calls += 1
            if fx_new < fx:
                x, fx = x_new, fx_new
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # If the step didn't help, try a smaller one or just keep x
                # Try half step
                if calls < budget:
                    x_new2 = clip(x - 0.25 * grad)
                    fx_new2 = func(x_new2)
                    calls += 1
                    if fx_new2 < fx:
                        x, fx = x_new2, fx_new2
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

        # Phase 2: Custom Nelder-Mead
        # Start simplex around best_x
        simplex = []
        simplex.append(best_x.copy())
        simplex_f = [best_f]
        
        # Initialize other vertices
        delta = 0.5 # Initial simplex size
        for i in range(dim):
            if calls >= budget:
                break
            v = best_x.copy()
            v[i] += delta
            v = clip(v)
            simplex.append(v)
            fv = func(v)
            calls += 1
            simplex_f.append(fv)
            if fv < best_f:
                best_f = fv
                best_x = v.copy()

        # Nelder-Mead Parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink
        tol = 1e-12
        max_iter = budget - calls

        for _ in range(max_iter):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            
            # Sort simplex
            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]
            
            # Check convergence
            if np.std(simplex_f) < tol and np.std([np.linalg.norm(v) for v in simplex]) < tol:
                break
                
            centroid = np.mean(simplex[:-1], axis=0)
            x_worst = simplex[-1]
            f_worst = simplex_f[-1]
            
            # Reflection
            x_r = centroid + alpha * (centroid - x_worst)
            x_r = clip(x_r)
            f_r = func(x_r)
            calls += 1
            if f_r < best_f:
                best_f = f_r
                best_x = x_r.copy()
            
            if f_r < simplex_f[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = clip(x_e)
                f_e = func(x_e)
                calls += 1
                if f_e < best_f:
                    best_f = f_e
                    best_x = x_e.copy()
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                # Contraction
                if f_r < f_worst:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                    x_c = clip(x_c)
                    f_c = func(x_c)
                    calls += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                    
                    if f_c < f_r:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = centroid + sigma * (simplex[i] - centroid)
                            simplex[i] = clip(simplex[i])
                            if calls >= budget:
                                break
                            f_s = func(simplex[i])
                            calls += 1
                            simplex_f[i] = f_s
                            if f_s < best_f:
                                best_f = f_s
                                best_x = simplex[i].copy()
                        if calls >= budget:
                            break
                else:
                    # Inside contraction
                    x_c = centroid + rho * (x_worst - centroid)
                    x_c = clip(x_c)
                    f_c = func(x_c)
                    calls += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                    
                    if f_c < f_worst:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = centroid + sigma * (simplex[i] - centroid)
                            simplex[i] = clip(simplex[i])
                            if calls >= budget:
                                break
                            f_s = func(simplex[i])
                            calls += 1
                            simplex_f[i] = f_s
                            if f_s < best_f:
                                best_f = f_s
                                best_x = simplex[i].copy()
                        if calls >= budget:
                            break

        # Phase 3: Final Polish with (1+1)-ES if budget remains
        if calls < budget and time.time() < deadline:
            x_local = best_x.copy()
            fx_local = best_f
            sigma = 0.01
            rng = np.random.default_rng(42)
            
            while calls < budget and time.time() < deadline:
                y = clip(x_local + sigma * rng.standard_normal(dim))
                fy = func(y)
                calls += 1
                
                if fy < fx_local:
                    x_local, fx_local = y, fy
                    sigma *= 1.1
                    if fx_local < best_f:
                        best_f = fx_local
                        best_x = x_local.copy()
                else:
                    sigma *= 0.7
                
                if sigma < 1e-10:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
