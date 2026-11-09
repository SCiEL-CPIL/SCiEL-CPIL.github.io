import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 290.0  # Safety margin before 300s kill
        dim = self.dim
        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return best_f
            if time.time() - start_time > time_limit:
                return best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            v = func(x)
            calls += 1
            if v < best_f:
                best_f = v
                best_x = x.copy()
            return v

        def check_time():
            return time.time() - start_time < time_limit - 2.0

        def check_budget():
            return calls < self.budget - 5

        # Phase 1: Diverse initialization
        # Use a mix of structured and random points
        init_points = []
        
        # Origin and near-origin points
        init_points.append(np.zeros(dim))
        for scale in [0.1, 0.5, 1.0]:
            for d in range(dim):
                p = np.zeros(dim)
                p[d] = scale
                p[d+1 if d+1 < dim else 0] = -scale
                init_points.append(p)
        
        # LHS points
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=dim, seed=123)
            lhs_points = sampler.random(n=30)
            lhs_points = lhs_points * 8.0 - 4.0  # Scale to [-4, 4]
            init_points.extend(lhs_points)
        except ImportError:
            init_points.extend(rng.uniform(-4, 4, (30, dim)))
        
        # Some random points for diversity
        init_points.extend(rng.uniform(-4.5, 4.5, (15, dim)))
        
        # Deduplicate approximately
        init_points = np.array(init_points)
        unique_mask = np.ones(len(init_points), dtype=bool)
        for i in range(len(init_points)):
            if not unique_mask[i]:
                continue
            for j in range(i+1, len(init_points)):
                if unique_mask[j] and np.linalg.norm(init_points[i] - init_points[j]) < 0.1:
                    unique_mask[j] = False
        init_points = init_points[unique_mask]
        
        # Evaluate all init points
        init_values = []
        for p in init_points:
            if not check_budget() or not check_time():
                break
            v = wrapped_func(p)
            init_values.append(v)
        
        if not init_values:
            return best_x, best_f
        
        # Sort by fitness
        sorted_idx = np.argsort(init_values)
        
        # Phase 2: Multi-start (1+1)-ES with adaptive sigma
        # Success-rate-based adaptation
        success_rate_target = 0.2
        sigma = 0.8
        sigma_min = 1e-6
        sigma_max = 5.0
        
        # Process top seeds with decreasing budget
        n_seeds = min(5, len(sorted_idx))
        budgets_per_seed = [180, 120, 80, 50, 30]
        
        for seed_i in range(n_seeds):
            if not check_budget() or not check_time():
                break
            
            x = init_points[sorted_idx[seed_i]].copy()
            f_x = init_values[sorted_idx[seed_i]]
            
            # Reset sigma for each seed, start larger for exploration
            sigma = max(0.3, 1.5 - 0.2 * seed_i)
            no_improve = 0
            max_no_improve = 40
            success_count = 0
            total_count = 0
            last_sigma = sigma
            
            # Adaptive step: use success rate to adjust sigma
            for step in range(budgets_per_seed[seed_i]):
                if not check_budget() or not check_time():
                    break
                if sigma < sigma_min:
                    break
                
                # Generate mutant
                mutation = rng.normal(0, 1, dim)
                y = x + sigma * mutation
                y = np.clip(y, -5, 5)
                
                f_y = wrapped_func(y)
                total_count += 1
                
                if f_y < f_x - 1e-12:
                    x = y
                    f_x = f_y
                    success_count += 1
                    no_improve = 0
                    # Increase sigma slightly on success
                    sigma = min(sigma * 1.02, sigma_max)
                else:
                    no_improve += 1
                    # Decrease sigma on failure
                    sigma = max(sigma * 0.95, sigma_min)
                
                # Periodically adjust sigma based on success rate
                if total_count % 20 == 0 and total_count >= 20:
                    sr = success_count / total_count
                    if sr > success_rate_target * 1.5:
                        sigma = min(sigma * 1.1, sigma_max)
                    elif sr < success_rate_target * 0.5:
                        sigma = max(sigma * 0.8, sigma_min)
                    success_count = 0
                    total_count = 0
                
                # Stagnation detection: restart from next-best seed
                if no_improve >= max_no_improve:
                    break
            
            # If we improved significantly, continue polishing
            if f_x < best_f - 1e-10:
                best_f = f_x
                best_x = x.copy()
        
        # Phase 3: Final high-precision polish on best point
        if best_x is not None and check_budget() and check_time():
            # Use a finer-grained local search
            x = best_x.copy()
            f_x = best_f
            sigma = 0.05
            no_improve = 0
            max_polish_steps = min(50, self.budget - calls - 5)
            
            for step in range(max_polish_steps):
                if not check_budget() or not check_time():
                    break
                if sigma < 1e-8:
                    break
                
                mutation = rng.normal(0, 1, dim)
                y = x + sigma * mutation
                y = np.clip(y, -5, 5)
                
                f_y = wrapped_func(y)
                
                if f_y < f_x - 1e-14:
                    x = y
                    f_x = f_y
                    sigma = min(sigma * 1.01, 0.1)
                    no_improve = 0
                else:
                    sigma = max(sigma * 0.9, 1e-8)
                    no_improve += 1
                    if no_improve > 30:
                        break
            
            if f_x < best_f:
                best_f = f_x
                best_x = x.copy()
        
        # Phase 4: Try scipy Nelder-Mead if time and budget allow
        if best_x is not None and check_budget() and check_time():
            try:
                from scipy.optimize import minimize
                # Use a wrapper that respects our call budget
                nm_calls = [0]
                def nm_wrapped(x):
                    nm_calls[0] += 1
                    if nm_calls[0] > 30 or not check_budget() or not check_time():
                        return best_f
                    return wrapped_func(x)
                
                res = minimize(nm_wrapped, best_x, method='Nelder-Mead',
                               options={'maxiter': 40, 'xatol': 1e-10, 'fatol': 1e-14,
                                        'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass
        
        return best_x, best_f
