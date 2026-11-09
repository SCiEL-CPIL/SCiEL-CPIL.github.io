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

        # Phase 1: Initial sampling - use a mix of deterministic and random points
        # For f13 (Sharp Ridge), the function has a ridge structure
        # We want to find points near the ridge quickly
        
        # Start with a few points along axes and diagonals
        init_points = []
        
        # Origin
        init_points.append(np.zeros(n))
        
        # Points along each axis
        for i in range(n):
            for sign in [1, -1]:
                x = np.zeros(n)
                x[i] = sign * 3.0
                init_points.append(x)
        
        # Diagonal points
        for sign in [1, -1]:
            x = np.ones(n) * sign * 2.0
            init_points.append(x)
            
        # Some random points
        for _ in range(10):
            init_points.append(rng.uniform(-4.0, 4.0, n))
        
        # Evaluate initial points
        for x in init_points:
            if calls >= budget:
                break
            try_func(x)
        
        if calls >= budget:
            return best_x, best_f

        # Phase 2: Coordinate descent with adaptive step sizes
        # This exploits the ridge structure by optimizing each coordinate
        # while maintaining momentum along the ridge direction
        
        # Initialize step sizes
        step_sizes = np.ones(n) * 1.0
        # Momentum for each coordinate
        momentum = np.zeros(n)
        # Track successful steps
        success_count = np.zeros(n)
        
        # Main coordinate descent loop
        max_cd_iterations = 200
        cd_iter = 0
        
        while cd_iter < max_cd_iterations and calls < budget and time.time() < deadline:
            cd_iter += 1
            improved = False
            
            for i in range(n):
                if calls >= budget or time.time() > deadline:
                    break
                    
                # Try positive and negative steps
                for direction in [1, -1]:
                    x_new = best_x.copy()
                    x_new[i] = best_x[i] + direction * step_sizes[i]
                    x_new = np.clip(x_new, -5.0, 5.0)
                    
                    f_new = try_func(x_new)
                    
                    if f_new < best_f:
                        # Success: move and increase step size
                        best_x = x_new
                        best_f = f_new
                        improved = True
                        # Adaptive step size: increase if we keep succeeding
                        success_count[i] += 1
                        if success_count[i] >= 3:
                            step_sizes[i] *= 1.5
                            success_count[i] = 0
                    else:
                        # Failure: decrease step size
                        success_count[i] = 0
                        step_sizes[i] *= 0.5
                        # Don't break - try the other direction
                        break  # Only try one direction per coordinate per iteration
                
                # Update momentum
                if improved:
                    momentum[i] = momentum[i] * 0.5 + direction * step_sizes[i] * 0.5
                else:
                    momentum[i] *= 0.5
                
                # Check if step size is too small
                if step_sizes[i] < 1e-6:
                    step_sizes[i] = 1e-6
            
            # If no improvement in a full cycle, try a small random perturbation
            if not improved:
                # Random restart in a small neighborhood
                x_perturb = best_x + rng.normal(0, 0.1, n)
                x_perturb = np.clip(x_perturb, -5.0, 5.0)
                f_perturb = try_func(x_perturb)
                if f_perturb < best_f:
                    best_x = x_perturb
                    best_f = f_perturb
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3: Ridge detection and search along the ridge
        # Estimate the ridge direction by finding the direction of least curvature
        
        # Sample points around the best point to estimate curvature
        eps = 0.1
        directions = []
        curvatures = []
        
        for _ in range(15):
            d = rng.standard_normal(n)
            d /= np.linalg.norm(d)
            
            f_plus = try_func(best_x + eps * d)
            f_minus = try_func(best_x - eps * d)
            
            # Approximate second derivative
            curvature = (f_plus - 2 * best_f + f_minus) / (eps ** 2)
            directions.append(d)
            curvatures.append(curvature)
        
        if len(curvatures) > 0:
            # Find the direction with minimum curvature (ridge direction)
            min_idx = np.argmin(curvatures)
            ridge_dir = directions[min_idx]
            
            # Search along the ridge direction
            # First, find a good interval
            t_left = 0.0
            t_right = 0.0
            f_left = best_f
            f_right = best_f
            
            # Expand left
            step = 0.5
            while calls < budget and time.time() < deadline:
                t_left -= step
                x_left = best_x + t_left * ridge_dir
                x_left = np.clip(x_left, -5.0, 5.0)
                f_left = try_func(x_left)
                if f_left < best_f:
                    best_f = f_left
                    best_x = x_left
                else:
                    break
                step *= 1.5
            
            # Expand right
            step = 0.5
            while calls < budget and time.time() < deadline:
                t_right += step
                x_right = best_x + t_right * ridge_dir
                x_right = np.clip(x_right, -5.0, 5.0)
                f_right = try_func(x_right)
                if f_right < best_f:
                    best_f = f_right
                    best_x = x_right
                else:
                    break
                step *= 1.5
            
            # Golden section search along the ridge
            if calls < budget and time.time() < deadline:
                def f_t(t):
                    x = best_x + t * ridge_dir
                    x = np.clip(x, -5.0, 5.0)
                    return try_func(x)
                
                gr = (np.sqrt(5) - 1) / 2
                a = t_left
                b = t_right
                c = b - gr * (b - a)
                d = a + gr * (b - a)
                fc = f_t(c)
                fd = f_t(d)
                
                while (b - a) > 1e-5 and calls < budget and time.time() < deadline:
                    if fc < fd:
                        b = d
                        d = c
                        fd = fc
                        c = b - gr * (b - a)
                        fc = f_t(c)
                    else:
                        a = c
                        c = d
                        fc = fd
                        d = a + gr * (b - a)
                        fd = f_t(d)
                
                t_mid = (a + b) / 2
                x_mid = best_x + t_mid * ridge_dir
                x_mid = np.clip(x_mid, -5.0, 5.0)
                f_mid = try_func(x_mid)
                if f_mid < best_f:
                    best_f = f_mid
                    best_x = x_mid

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

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
