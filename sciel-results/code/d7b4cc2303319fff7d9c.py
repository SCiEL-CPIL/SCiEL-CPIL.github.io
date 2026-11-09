import scipy
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        calls = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls >= budget:
                raise StopIteration("Budget exceeded")
            
            x = np.asarray(x)
            f = func(x)
            calls += 1
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Multi-point Global Search with (1+1)-ES and Restarts
        # Use 55% of budget for global exploration to leave more room for local refinement
        global_budget = int(budget * 0.55)
        
        # Multi-point initialization for better early coverage
        n_init = min(5, global_budget // 10)
        trajectories = []
        
        for i in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            x_init = rng.uniform(-5, 5, dim)
            try:
                fx_init = safe_func(x_init)
            except (TimeoutError, StopIteration):
                break
            trajectories.append({'x': x_init.copy(), 'fx': fx_init, 'sigma': 1.5 + 0.5 * i})
        
        # If no trajectories were created (shouldn't happen), create one
        if not trajectories:
            x_init = rng.uniform(-5, 5, dim)
            try:
                fx_init = safe_func(x_init)
            except (TimeoutError, StopIteration):
                return best_x, best_f
            trajectories.append({'x': x_init.copy(), 'fx': fx_init, 'sigma': 1.5})
        
        # Run (1+1)-ES on each trajectory until budget or deadline
        min_sigma = 0.005
        max_sigma = 8.0
        max_stagnation_per_traj = 20
        
        while calls < budget and time.time() < deadline:
            progressed = False
            for idx, traj in enumerate(trajectories):
                if calls >= budget or time.time() > deadline:
                    break
                
                x = traj['x']
                fx = traj['fx']
                sigma = traj['sigma']
                stagnation = 0
                
                # Run up to max_stagnation_per_traj steps on this trajectory
                for _ in range(max_stagnation_per_traj):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    y = x + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    
                    try:
                        fy = safe_func(y)
                    except (TimeoutError, StopIteration):
                        break
                    
                    if fy < fx:
                        x, fx = y, fy
                        sigma = min(sigma * 1.06, max_sigma)
                        stagnation = 0
                        progressed = True
                    else:
                        sigma = max(sigma * 0.93, min_sigma)
                        stagnation += 1
                        if stagnation >= max_stagnation_per_traj:
                            break
                    
                    # Update trajectory state
                    trajectories[idx]['x'] = x
                    trajectories[idx]['fx'] = fx
                    trajectories[idx]['sigma'] = sigma
                
                # If this trajectory stagnated, decide whether to restart it
                if stagnation >= max_stagnation_per_traj:
                    # Restart this trajectory from a diverse point
                    if best_x is not None:
                        r = rng.random()
                        if r < 0.25:
                            x = best_x.copy()
                            fx = best_f
                        elif r < 0.6:
                            x = best_x + 2.5 * rng.standard_normal(dim)
                        else:
                            x = rng.uniform(-5, 5, dim)
                        x = np.clip(x, -5, 5)
                        try:
                            fx = safe_func(x)
                        except (TimeoutError, StopIteration):
                            break
                    else:
                        x = rng.uniform(-5, 5, dim)
                        try:
                            fx = safe_func(x)
                        except (TimeoutError, StopIteration):
                            break
                    
                    trajectories[idx]['x'] = x
                    trajectories[idx]['fx'] = fx
                    trajectories[idx]['sigma'] = 2.0
                    progressed = True
            
            # If no trajectory made progress in a full cycle, we've likely converged
            if not progressed:
                break

        # Phase 2: Local Refinement using Nelder-Mead
        # Nelder-Mead is derivative-free and robust for non-smooth functions like Schwefel
        if calls < budget and time.time() < deadline and best_x is not None:
            remaining_calls = budget - calls
            if remaining_calls > 15:
                # Nelder-Mead: use a fraction of remaining budget
                # Each iteration uses ~dim+1 calls, so max_iter should be conservative
                max_iter = min(remaining_calls // (dim + 2), 50)
                max_iter = max(8, max_iter)
                
                try:
                    # Initialize simplex around best_x with reasonable spread
                    x0 = best_x.copy()
                    
                    result = scipy.optimize.minimize(
                        safe_func, 
                        x0, 
                        method='Nelder-Mead', 
                        options={
                            'maxiter': max_iter,
                            'xatol': 1e-12,
                            'fatol': 1e-12,
                            'adaptive': True,
                            'initial_simplex': None
                        }
                    )
                except (TimeoutError, StopIteration, Exception):
                    pass
        
        # Phase 3: Final High-Precision Polish with CMA-ES-inspired adaptive (1+1)-ES
        # Uses success-rate-based step size adaptation for better convergence
        if calls < budget and time.time() < deadline and best_x is not None:
            remaining_calls = budget - calls
            if remaining_calls > 5:
                x = best_x.copy()
                fx = best_f  # best_f is already the value at best_x
                
                # CMA-ES-inspired: track success rate for step size adaptation
                sigma = 0.03
                success_window = []
                window_size = 8
                target_success_rate = 0.15
                
                for _ in range(remaining_calls):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    y = x + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    
                    try:
                        fy = safe_func(y)
                    except (TimeoutError, StopIteration):
                        break
                    
                    success = fy < fx
                    success_window.append(1 if success else 0)
                    if len(success_window) > window_size:
                        success_window.pop(0)
                    
                    if success:
                        x, fx = y, fy
                    
                    # Adapt sigma based on success rate over recent window
                    if len(success_window) >= window_size:
                        recent_rate = sum(success_window) / len(success_window)
                        if recent_rate > target_success_rate:
                            sigma = min(sigma * 1.04, 0.5)
                        elif recent_rate < target_success_rate * 0.4:
                            sigma = max(sigma * 0.88, 1e-7)
                        else:
                            sigma = max(sigma * 0.97, 1e-7)
                    
                    # If sigma becomes very small, we've likely converged
                    if sigma < 1e-6:
                        break

        return best_x, best_f
