import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0
        
        n = self.dim
        calls = 0
        best_x = None
        best_f = np.inf
        
        rng = np.random.default_rng(42)
        
        def try_eval(x):
            nonlocal calls, best_f, best_x
            if calls >= self.budget or time.time() > deadline:
                return None
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        def golden_section_line_search(x, direction, f_x):
            """Find minimum along direction from x using golden section search."""
            nonlocal calls
            # First find a bracket
            step = 0.1
            a = -step
            b = step
            f_a = f_x
            f_b = None
            
            # Try positive direction first
            for _ in range(10):
                if time.time() > deadline or calls >= self.budget:
                    return x, f_x
                x_b = np.clip(x + b * direction, -5, 5)
                f_b = try_eval(x_b)
                if f_b is None:
                    return x, f_x
                if f_b < f_x:
                    break
                a = b
                b *= 2
                f_a = f_b
            
            if f_b is None or f_b >= f_x:
                # Try negative direction
                step = 0.1
                a = step
                b = -step
                f_a = f_x
                f_b = None
                for _ in range(10):
                    if time.time() > deadline or calls >= self.budget:
                        return x, f_x
                    x_b = np.clip(x + b * direction, -5, 5)
                    f_b = try_eval(x_b)
                    if f_b is None:
                        return x, f_x
                    if f_b < f_x:
                        break
                    b = -a
                    a = -b
                    f_a = f_b
                if f_b is None or f_b >= f_x:
                    return x, f_x
            
            # Ensure a < b
            if a > b:
                a, b = b, a
                f_a, f_b = f_b, f_a
            
            # Golden section search
            gr = (np.sqrt(5) - 1) / 2
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            x_c = np.clip(x + c * direction, -5, 5)
            x_d = np.clip(x + d * direction, -5, 5)
            f_c = try_eval(x_c)
            f_d = try_eval(x_d)
            if f_c is None or f_d is None:
                return x, f_x
            
            for _ in range(40):
                if time.time() > deadline or calls >= self.budget:
                    break
                if f_c < f_d:
                    b = d
                    f_b = f_d
                    d = c
                    f_d = f_c
                    c = b - gr * (b - a)
                    x_d = np.clip(x + c * direction, -5, 5)
                    f_c = try_eval(x_d)
                    if f_c is None:
                        break
                else:
                    a = c
                    f_a = f_c
                    c = d
                    f_c = f_d
                    d = a + gr * (b - a)
                    x_c = np.clip(x + d * direction, -5, 5)
                    f_d = try_eval(x_c)
                    if f_d is None:
                        break
                if abs(b - a) < 1e-7:
                    break
            
            # Return best point
            if f_c < f_d:
                return x_c, f_c
            else:
                return x_d, f_d
        
        def powell_method(x0, max_cycles=8):
            """Powell's conjugate direction method."""
            nonlocal calls, best_f, best_x
            x = x0.copy()
            f_x = try_eval(x)
            if f_x is None:
                return x, np.inf
            
            directions = np.eye(n)
            
            for cycle in range(max_cycles):
                if time.time() > deadline or calls >= self.budget:
                    break
                
                x_start = x.copy()
                f_start = f_x
                
                for i in range(n):
                    if time.time() > deadline or calls >= self.budget:
                        break
                    x, f_x = golden_section_line_search(x, directions[i], f_x)
                
                new_dir = x - x_start
                if np.linalg.norm(new_dir) < 1e-10:
                    break
                
                # Check improvement
                f_new = try_eval(np.clip(x + new_dir, -5, 5))
                if f_new is None:
                    break
                
                if f_new < f_start - 1e-10:
                    directions[n-1] = new_dir / np.linalg.norm(new_dir)
                    x = x + new_dir
                    f_x = f_new
                elif abs(f_new - f_start) < 1e-8:
                    break
            
            return x, f_x
        
        def nelder_mead(x0, max_iter=150):
            """Nelder-Mead simplex method."""
            nonlocal calls, best_f, best_x
            x = x0.copy()
            f_x = try_eval(x)
            if f_x is None:
                return x, np.inf
            
            simplex = [x.copy()]
            f_simplex = [f_x]
            for i in range(n):
                xi = x.copy()
                xi[i] += 0.5 if xi[i] < 0 else -0.5
                xi = np.clip(xi, -5, 5)
                fi = try_eval(xi)
                if fi is None:
                    return x, f_x
                simplex.append(xi)
                f_simplex.append(fi)
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            for _ in range(max_iter):
                if time.time() > deadline or calls >= self.budget:
                    break
                
                order = np.argsort(f_simplex)
                simplex = [simplex[i].copy() for i in order]
                f_simplex = [f_simplex[i] for i in order]
                
                if abs(f_simplex[0] - f_simplex[-1]) < 1e-12:
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                x_r = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                f_r = try_eval(x_r)
                if f_r is None:
                    break
                
                if f_r < f_simplex[0]:
                    x_e = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                    f_e = try_eval(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                elif f_r < f_simplex[-2]:
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    if f_r < f_simplex[-1]:
                        x_c = np.clip(centroid + rho * (x_r - centroid), -5, 5)
                    else:
                        x_c = np.clip(centroid - rho * (simplex[-1] - centroid), -5, 5)
                    f_c = try_eval(x_c)
                    if f_c is None:
                        break
                    if f_c < min(f_r, f_simplex[-1]):
                        simplex[-1] = x_c
                        f_simplex[-1] = f_c
                    else:
                        for i in range(1, n+1):
                            simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), -5, 5)
                            fi = try_eval(simplex[i])
                            if fi is None:
                                return x, f_x
                            f_simplex[i] = fi
            
            idx = np.argmin(f_simplex)
            return simplex[idx], f_simplex[idx]
        
        # Phase 1: Multiple Powell starts
        n_starts = 4
        for s in range(n_starts):
            if time.time() > deadline or calls >= self.budget:
                break
            if s == 0:
                x0 = rng.uniform(-2, 2, n)
            else:
                x0 = rng.uniform(-4, 4, n)
            powell_method(x0, max_cycles=6)
        
        # Phase 2: Polish with Nelder-Mead
        if best_x is not None and calls < self.budget and time.time() < deadline:
            nelder_mead(best_x, max_iter=120)
        
        if best_x is None:
            best_x = np.zeros(n)
            f_val = try_eval(best_x)
            if f_val is not None:
                best_f = f_val
        
        return best_x, best_f
