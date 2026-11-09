import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 295.0  # Safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() - start_time < time_limit

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            if not time_ok():
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                v = float(func(x))
            except Exception:
                return best_f
            if not np.isfinite(v):
                v = best_f
                return v
            calls += 1
            if v < best_f:
                best_f = v
                best_x = x.copy()
            return v

        # 1. Initialization with Latin Hypercube + Origin + Sparse Grid
        init_points = []
        init_points.append(np.zeros(dim))
        
        # Add a few points at the center of each dimension
        for d in range(dim):
            p = np.zeros(dim)
            p[d] = 0.0
            init_points.append(p)
        
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=dim, seed=42)
            points = sampler.random(n=25)
            # Scale from [0,1] to [-4.5,4.5] for better coverage
            points = points * 9.0 - 4.5
            init_points.extend(points)
        except ImportError:
            init_points.extend(rng.uniform(-4.5, 4.5, (25, dim)))
        
        # Evaluate init
        init_values = []
        for p in init_points:
            if not time_ok() or calls >= budget - 80:
                break
            v = wrapped_func(p)
            init_values.append(v)
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped_func(best_x)
        
        sorted_idx = np.argsort(init_values)
        # Get top 3 starting points for multi-start (fewer but better for CMA-ES)
        top_starts = []
        for idx in sorted_idx[:3]:
            top_starts.append(init_points[idx].copy())

        # 2. Multi-Start CMA-ES with proper fitness tracking
        use_cma = False
        try:
            import cma
            use_cma = True
        except ImportError:
            use_cma = False

        if use_cma:
            # Allocate budget for CMA-ES
            # CMA-ES is more efficient for multimodal problems when given enough budget per run
            # Use 3 restarts with the best starting points
            n_starts = 3
            remaining_budget = budget - calls - 20
            if remaining_budget < 100:
                n_starts = 1
            elif remaining_budget < 200:
                n_starts = 2
            
            for i in range(n_starts):
                if not time_ok() or calls >= budget - 10:
                    break
                
                x0 = top_starts[i].copy() if i < len(top_starts) else best_x.copy()
                
                # CMA-ES parameters - small popsize for budget efficiency
                sigma0 = 1.5 if i == 0 else 0.8
                bounds = [(-5.0, 5.0)] * dim
                popsize = max(4, int(4 + 2 * np.log(dim)))
                
                try:
                    es = cma.CMA(x0, sigma0, {
                        'bounds': bounds,
                        'maxiter': 500,
                        'seed': i + 100,
                        'verbose': -9,
                        'tolopt': 1e-14,
                        'tolfun': 1e-14,
                        'tolfunint': 1e-14,
                        'popsize': popsize,
                        'ccov': 2.0 / (dim + 4),  # Slightly faster covariance adaptation
                        'cmean': 1.0 / (dim + 6),  # Slightly faster mean adaptation
                    })
                    
                    while not es.stop() and calls < budget - 10 and time_ok():
                        solutions = es.ask()
                        
                        # Evaluate all solutions and capture fitness values
                        fitness_values = []
                        valid_count = 0
                        for sol in solutions:
                            if calls >= budget - 10 or not time_ok():
                                break
                            # wrapped_func returns the function value
                            f_val = wrapped_func(sol)
                            fitness_values.append(f_val)
                            valid_count += 1
                        
                        if valid_count == 0:
                            break
                        
                        # Tell CMA-ES the fitness values
                        if valid_count < len(solutions):
                            # If we couldn't evaluate all, break
                            break
                        
                        es.tell(solutions, fitness_values)
                        
                except Exception:
                    pass
            # CMA-ES has been run, skip to final polish
            if best_x is not None and calls < budget - 5 and time_ok():
                # Run a tight Nelder-Mead on the best found point
                try:
                    from scipy.optimize import minimize
                    res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                                   options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-14, 'maxfev': 50})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass
            return best_x, best_f
            
        # 2. Fallback: Multi-Start (1+1)-ES if CMA not available
        if not use_cma:
            step_budgets = [150, 100, 80]
            
            for i, x0 in enumerate(top_starts):
                if not time_ok() or calls >= budget - 10:
                    break
                
                x = x0.copy()
                f_x = init_values[sorted_idx[i]]
                
                sigma = 0.8
                no_improve = 0
                max_steps = step_budgets[i]
                
                for _ in range(max_steps):
                    if not time_ok() or calls >= budget - 5:
                        break
                    if sigma < 1e-8:
                        break
                        
                    # Generate mutant with occasional larger jumps for multimodal landscape
                    if rng.random() < 0.1:
                        mutation = rng.normal(0, 3, dim)
                    else:
                        mutation = rng.normal(0, 1, dim)
                    y = x + sigma * mutation
                    y = np.clip(y, -5, 5)
                    
                    f_y = wrapped_func(y)
                    
                    if f_y < f_x - 1e-10:
                        x = y
                        f_x = f_y
                        sigma *= 1.08
                        no_improve = 0
                    else:
                        sigma *= 0.92
                        no_improve += 1
                        if no_improve > 40:
                            break
            
            # 3. Final Polish
            if best_x is not None and calls < budget - 5 and time_ok():
                try:
                    from scipy.optimize import minimize
                    res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                                   options={'maxiter': 80, 'xatol': 1e-10, 'fatol': 1e-14, 'maxfev': 40})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        return best_x, best_f
