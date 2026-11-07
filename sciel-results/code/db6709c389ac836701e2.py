import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        
        # Global best tracking
        global_best_x = None
        global_best_f = np.inf
        total_calls = 0
        
        def track(x, f):
            nonlocal global_best_x, global_best_f
            if f < global_best_f:
                global_best_f = f
                global_best_x = x.copy()
        
        def safe_func(x):
            nonlocal total_calls
            if total_calls >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            total_calls += 1
            return func(np.clip(x, -5, 5))
        
        # Phase 1: Random initialization
        init_budget = min(30, budget)
        for i in range(init_budget):
            x = rng.uniform(-5, 5, d)
            f = safe_func(x)
            if f < np.inf:
                track(x, f)
        
        if global_best_x is None:
            x = np.zeros(d)
            f = safe_func(x)
            track(x, f)
            if global_best_x is None:
                return np.zeros(d), 0.0
        
        # Phase 2: Powell's Method
        def powell_method(x_start, f_start):
            x = x_start.copy()
            f = f_start
            
            # Initialize directions to unit vectors
            directions = np.eye(d).copy()
            
            min_step = 1e-10
            max_step = 10.0
            golden = (np.sqrt(5) - 1) / 2
            
            def line_search(x, direction, f_current):
                """
                Perform a 1D line search along 'direction' starting from 'x'.
                Returns (new_x, new_f)
                """
                # First, find an interval [a, b] that brackets the minimum
                step = 0.1
                a = 0.0
                b = step
                f_a = f_current
                f_b = safe_func(x + b * direction)
                
                if f_b == np.inf:
                    return x.copy(), f_current
                
                # Expand interval if needed
                max_expand = 8
                for _ in range(max_expand):
                    if f_b < f_a:
                        a = b
                        f_a = f_b
                        b = 2 * b
                        if b > max_step:
                            b = max_step
                            f_b = safe_func(x + b * direction)
                            break
                        f_b = safe_func(x + b * direction)
                        if f_b == np.inf:
                            break
                    else:
                        break
                
                # If we couldn't bracket, return the best found
                if f_b == np.inf:
                    if f_a < f_current:
                        return np.clip(x + a * direction, -5, 5), f_a
                    return x.copy(), f_current
                
                # Golden section search
                c = a + (1 - golden) * (b - a)
                dd = a + golden * (b - a)
                
                f_c = safe_func(x + c * direction)
                f_dd = safe_func(x + dd * direction)
                
                if f_c == np.inf and f_dd == np.inf:
                    # Both failed, return best known
                    best_f = min(f_a, f_b, f_current)
                    if best_f == f_a:
                        return np.clip(x + a * direction, -5, 5), f_a
                    elif best_f == f_b:
                        return np.clip(x + b * direction, -5, 5), f_b
                    return x.copy(), f_current
                
                max_iter = 25
                for _ in range(max_iter):
                    if b - a < min_step:
                        break
                    if f_c == np.inf:
                        # c is invalid, shrink toward dd
                        a = c
                        c = dd
                        f_c = f_dd
                        dd = a + golden * (b - a)
                        f_dd = safe_func(x + dd * direction)
                    elif f_dd == np.inf:
                        # dd is invalid, shrink toward c
                        b = dd
                        dd = c
                        f_dd = f_c
                        c = a + (1 - golden) * (b - a)
                        f_c = safe_func(x + c * direction)
                    elif f_c < f_dd:
                        b = dd
                        dd = c
                        f_dd = f_c
                        c = a + (1 - golden) * (b - a)
                        f_c = safe_func(x + c * direction)
                    else:
                        a = c
                        c = dd
                        f_c = f_dd
                        dd = a + golden * (b - a)
                        f_dd = safe_func(x + dd * direction)
                
                # Find the best point
                candidates = [(a, f_a), (c, f_c if f_c != np.inf else np.inf), 
                             (dd, f_dd if f_dd != np.inf else np.inf), (b, f_b)]
                valid_candidates = [(t, f) for t, f in candidates if f != np.inf]
                if not valid_candidates:
                    return x.copy(), f_current
                best_t, best_f = min(valid_candidates, key=lambda x: x[1])
                return np.clip(x + best_t * direction, -5, 5), best_f
            
            # Main Powell loop
            max_iterations = 50
            for iteration in range(max_iterations):
                if total_calls >= budget or time.time() > deadline:
                    break
                
                min_f_in_cycle = f
                min_x_in_cycle = x.copy()
                
                for i in range(d):
                    if total_calls >= budget or time.time() > deadline:
                        break
                    
                    x_new, f_new = line_search(x, directions[i], f)
                    
                    if f_new < min_f_in_cycle:
                        min_f_in_cycle = f_new
                        min_x_in_cycle = x_new
                    
                    x = x_new
                    f = f_new
                
                if total_calls >= budget or time.time() > deadline:
                    break
                
                # Check convergence
                if abs(f - min_f_in_cycle) < 1e-12:
                    break
                
                # Update directions: replace the direction that gave the worst improvement
                # with a new direction from the start of the cycle to the best point
                # For simplicity, we periodically re-orthogonalize
                
                if iteration > 0 and iteration % 10 == 0:
                    # Re-orthogonalize directions using QR decomposition
                    try:
                        Q, R = np.linalg.qr(directions)
                        # Ensure right-handed system
                        if np.linalg.det(Q) < 0:
                            Q[:, 0] *= -1
                        directions = Q
                    except:
                        pass
            
            return x, f
        
        x_powell, f_powell = powell_method(global_best_x, global_best_f)
        track(x_powell, f_powell)
        
        # Phase 3: If we have budget left, try a few more local searches from different starts
        if total_calls < budget - 10 and time.time() < deadline:
            # Try Nelder-Mead from the current best
            def nelder_mead(x_start, f_start, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0):
                x = x_start.copy()
                f = f_start
                
                # Create initial simplex
                simplex = [x.copy()]
                for i in range(d):
                    xi = x.copy()
                    step = 0.1 if x[i] == 0 else 0.1 * abs(x[i])
                    xi[i] += step
                    simplex.append(xi)
                
                simplex_f = [f] + [safe_func(s) for s in simplex[1:]]
                # Handle inf values
                simplex_f = [sf if sf != np.inf else f + 1e10 * (i+1) for i, sf in enumerate(simplex_f)]
                
                max_iter = 100
                for _ in range(max_iter):
                    if total_calls >= budget or time.time() > deadline:
                        break
                    
                    # Sort simplex by function value
                    idx = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in idx]
                    simplex_f = [simplex_f[i] for i in idx]
                    
                    # Check convergence
                    if abs(simplex_f[-1] - simplex_f[0]) < 1e-10:
                        break
                    
                    # Compute centroid of all but worst point
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    x_r = centroid + alpha * (centroid - simplex[-1])
                    f_r = safe_func(x_r)
                    if f_r == np.inf:
                        f_r = simplex_f[-1] + 1e10
                    
                    if simplex_f[0] <= f_r < simplex_f[-2]:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    elif f_r < simplex_f[0]:
                        # Expansion
                        x_e = centroid + gamma * (x_r - centroid)
                        f_e = safe_func(x_e)
                        if f_e == np.inf:
                            f_e = f_r + 1e10
                        if f_e < f_r:
                            simplex[-1] = x_e
                            simplex_f[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                    else:
                        # Contraction
                        if f_r < simplex_f[-1]:
                            # Outside contraction
                            x_c = centroid + 0.5 * (x_r - centroid)
                            f_c = safe_func(x_c)
                            if f_c == np.inf:
                                f_c = f_r + 1e10
                            if f_c <= min(f_r, simplex_f[-1]):
                                simplex[-1] = x_c
                                simplex_f[-1] = f_c
                            else:
                                # Shrink
                                for i in range(1, d+1):
                                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                    simplex_f[i] = safe_func(simplex[i])
                                    if simplex_f[i] == np.inf:
                                        simplex_f[i] = simplex_f[0] + 1e10 * i
                        else:
                            # Inside contraction
                            x_c = centroid - 0.5 * (simplex[-1] - centroid)
                            f_c = safe_func(x_c)
                            if f_c == np.inf:
                                f_c = simplex_f[-1] + 1e10
                            if f_c < simplex_f[-1]:
                                simplex[-1] = x_c
                                simplex_f[-1] = f_c
                            else:
                                # Shrink
                                for i in range(1, d+1):
                                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                    simplex_f[i] = safe_func(simplex[i])
                                    if simplex_f[i] == np.inf:
                                        simplex_f[i] = simplex_f[0] + 1e10 * i
                
                # Return best point
                best_idx = np.argmin(simplex_f)
                return simplex[best_idx], simplex_f[best_idx]
            
            x_nm, f_nm = nelder_mead(global_best_x, global_best_f)
            track(x_nm, f_nm)
        
        if global_best_x is not None:
            return global_best_x, global_best_f
        else:
            return np.zeros(d), 0.0
