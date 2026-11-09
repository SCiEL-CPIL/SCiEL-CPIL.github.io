from scipy.optimize import minimize
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        n = self.dim
        budget = self.budget

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        def try_func(x):
            nonlocal best_f, best_x, calls
            if calls >= budget or time.time() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Halton grid initialization for good coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i = i // base
            return r

        n_init = 20
        for i in range(n_init):
            x = np.array([halton(i + 1, base) for base in [2, 3, 5, 7, 11]])
            x = x * 10.0 - 5.0  # Map to [-5, 5]
            try_func(x)

        if calls >= budget:
            return best_x, best_f

        # Phase 2: Nelder-Mead simplex method adapted for sharp ridge
        # Initialize simplex around the best point found
        center = best_x.copy()
        simplex = [center.copy()]
        
        # Create simplex with adaptive initial step sizes
        for i in range(n):
            x_plus = center.copy()
            x_plus[i] += 0.5
            x_plus = np.clip(x_plus, -5.0, 5.0)
            simplex.append(x_plus)
        
        # Evaluate initial simplex
        f_values = []
        for i, x in enumerate(simplex):
            if i == 0:
                f_values.append(best_f)
            else:
                f_values.append(try_func(x))
        
        # Nelder-Mead parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrinkage
        max_nm_iter = 150
        nm_iter = 0
        
        while nm_iter < max_nm_iter and calls < budget and time.time() < deadline:
            nm_iter += 1
            
            # Sort simplex by function value
            order = np.argsort(f_values)
            simplex = [simplex[i] for i in order]
            f_values = [f_values[i] for i in order]
            
            # Check convergence
            if f_values[-1] - f_values[0] < 1e-10:
                break
            
            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = try_func(x_r)
            
            if f_values[0] <= f_r < f_values[-2]:
                # Accept reflection
                simplex[-1] = x_r
                f_values[-1] = f_r
            elif f_r < f_values[0]:
                # Try expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = try_func(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
            else:
                # Contraction
                if f_r < f_values[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (x_r - centroid)
                x_c = np.clip(x_c, -5.0, 5.0)
                f_c = try_func(x_c)
                
                if f_c < min(f_r, f_values[-1]):
                    simplex[-1] = x_c
                    f_values[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        f_values[i] = try_func(simplex[i])
            
            # Update best
            if f_values[0] < best_f:
                best_f = f_values[0]
                best_x = simplex[0].copy()
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3: Curvature-aligned line search along ridge
        # Estimate gradient and Hessian-free direction using finite differences
        eps = 0.05
        grad = np.zeros(n)
        for i in range(n):
            x_plus = best_x.copy()
            x_plus[i] += eps
            x_minus = best_x.copy()
            x_minus[i] -= eps
            x_plus = np.clip(x_plus, -5.0, 5.0)
            x_minus = np.clip(x_minus, -5.0, 5.0)
            f_plus = try_func(x_plus)
            f_minus = try_func(x_minus)
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f
        
        # Estimate curvature in gradient direction to find valley floor
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad_dir = grad / grad_norm
            
            # Find curvature along gradient direction
            t_test = 0.5
            x_test = best_x + t_test * grad_dir
            x_test = np.clip(x_test, -5.0, 5.0)
            f_test = try_func(x_test)
            
            # If moving in gradient direction doesn't help much, we're on the ridge
            # Try perpendicular directions to find the ridge axis
            if f_test > best_f + 0.01:
                # We're likely on the ridge - search perpendicular to gradient
                # Find a direction orthogonal to gradient with low curvature
                for attempt in range(10):
                    d = rng.standard_normal(n)
                    d -= np.dot(d, grad_dir) * grad_dir
                    d_norm = np.linalg.norm(d)
                    if d_norm < 1e-12:
                        continue
                    d /= d_norm
                    
                    # Check curvature in this direction
                    f_d_plus = try_func(best_x + eps * d)
                    f_d_minus = try_func(best_x - eps * d)
                    curvature = (f_d_plus - 2 * best_f + f_d_minus) / (eps ** 2)
                    
                    if curvature < 10:  # Low curvature = ridge direction
                        ridge_dir = d
                        break
                else:
                    ridge_dir = d
            else:
                # Gradient direction is good, use it
                ridge_dir = grad_dir
        
        # Line search along the identified direction
        if calls < budget and time.time() < deadline:
            # Golden section search
            a = -2.0
            b = 2.0
            gr = (np.sqrt(5) - 1) / 2
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            def f_line(t):
                x = best_x + t * ridge_dir
                x = np.clip(x, -5.0, 5.0)
                return try_func(x)
            
            fc = f_line(c)
            fd = f_line(d)
            
            while (b - a) > 1e-6 and calls < budget and time.time() < deadline:
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = f_line(c)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = f_line(d)
            
            t_best = (a + b) / 2
            x_best = best_x + t_best * ridge_dir
            x_best = np.clip(x_best, -5.0, 5.0)
            f_best = try_func(x_best)
            if f_best < best_f:
                best_f = f_best
                best_x = x_best

        # Phase 4: Final polish with L-BFGS-B
        if calls < budget - 5 and time.time() < deadline:
            def obj(x):
                return try_func(x)
            
            try:
                res = minimize(obj, best_x, method='L-BFGS-B', 
                              bounds=[(-5, 5)] * n,
                              options={'maxiter': 30, 'gtol': 1e-8, 'ftol': 1e-12})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        return best_x, best_f
