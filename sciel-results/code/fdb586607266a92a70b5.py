import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = 280.0  # Safety margin
        n = self.dim
        
        best_x = np.zeros(n)
        best_f = np.inf
        total_evals = 0
        rng = np.random.default_rng(42)

        def safe_func(x):
            nonlocal total_evals
            if time.time() - start_time > deadline:
                raise Exception("Time limit exceeded")
            if not isinstance(x, np.ndarray) or x.shape != (n,) or not np.all(np.isfinite(x)):
                return 1e10
            total_evals += 1
            if total_evals > self.budget:
                raise Exception("Budget exceeded")
            try:
                val = float(func(x))
                if not np.isfinite(val):
                    return 1e10
                return val
            except Exception:
                return 1e10

        # Custom Fast Nelder-Mead
        def nelder_mead(func, x0, max_iter=500):
            # Initialize simplex
            # x0 is the best vertex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(1, n + 1):
                simplex[i] = x0.copy()
                # Expand along axis i-1
                simplex[i, i - 1] += 0.05 if x0[i - 1] == 0 else 0.05 * abs(x0[i - 1])
                # If x0 is 0, use a small positive step to stay in domain if needed, 
                # but NM handles bounds via reflection usually. 
                # For f6, we want to explore the sector.
                if x0[i-1] == 0:
                     simplex[i, i-1] = 0.05

            # Evaluate initial simplex
            fvals = np.array([func(simplex[i]) for i in range(n + 1)])
            
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrink

            for _ in range(max_iter):
                # Sort simplex by fvals
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                # Check convergence
                if np.ptp(fvals) < 1e-10:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = func(x_r)
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = func(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Simple reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outer contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inner contraction
                        x_c = centroid - rho * (x_r - centroid)
                    
                    f_c = func(x_c)
                    
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
            
            # Return best point
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]

        # Multi-start strategy
        # 1. Origin
        # 2. Points biased towards the "Attractive Sector" (x1 > 0)
        # The sector is defined by x1 > 0 and ||x_2:n|| < x1.
        # We generate starts that satisfy this condition to help NM find the global min faster.
        
        starts = [np.zeros(n)]
        
        # Generate random points inside the sector
        for _ in range(10):
            x1 = rng.uniform(0.5, 5.0)
            # Sample x_2:n such that ||x_2:n|| < x1
            # Sample a random direction in n-1 dims
            dir_vec = rng.normal(size=n-1)
            norm_dir = np.linalg.norm(dir_vec)
            if norm_dir == 0:
                dir_vec = np.zeros(n-1)
                dir_vec[0] = 1
            else:
                dir_vec /= norm_dir
            
            # Random radius r < x1
            r = rng.uniform(0, x1 * 0.9)
            x_tail = r * dir_vec
            
            x_start = np.zeros(n)
            x_start[0] = x1
            x_start[1:] = x_tail
            starts.append(x_start)
            
        # Also some random general starts
        for _ in range(5):
            starts.append(rng.uniform(-5, 5, n))

        # Shuffle starts to mix quality
        rng.shuffle(starts)

        for x0 in starts:
            if time.time() - start_time > deadline:
                break
            if total_evals >= self.budget:
                break
            
            # Estimate max iterations
            remaining = self.budget - total_evals
            # NM uses ~n+1 evals per init, then 1-2 per iter. 
            # Be conservative.
            max_iter = max(50, remaining // 10) 
            # Actually, let's just run until budget or time.
            # But we need to pass max_iter to prevent infinite loops if stuck.
            
            try:
                x_opt, f_opt = nelder_mead(safe_func, x0, max_iter=max_iter)
                if f_opt < best_f:
                    best_f = f_opt
                    best_x = x_opt.copy()
            except Exception:
                # If budget exceeded or time, break
                break

        return best_x, best_f
