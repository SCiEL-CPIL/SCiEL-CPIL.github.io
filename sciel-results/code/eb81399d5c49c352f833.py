import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        calls = 0
        best_x = None
        best_f = np.inf
        
        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls >= self.budget:
                raise StopIteration("Budget exceeded")
            
            x = np.asarray(x, dtype=np.float64)
            f = func(x)
            calls += 1
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)
        
        # Phase 1: Global Search with (1+1)-ES and Restarts
        # Use 60% of budget for global exploration
        global_budget = int(self.budget * 0.6)
        
        # (1+1)-ES with window-based success-rate sigma adaptation
        x = rng.uniform(-0.5, 0.5, self.dim)
        sigma = 2.0
        sigma_min = 1e-4
        sigma_max = 4.0
        stall_count = 0
        stall_limit = 40
        restart_count = 0
        max_restarts = 8
        window_size = 20
        success_window = []
        
        # First evaluate the starting point
        try:
            fx = safe_func(x)
        except (TimeoutError, StopIteration):
            return best_x, best_f
        
        while calls < global_budget and time.time() < deadline:
            # Generate candidate
            direction = rng.standard_normal(self.dim)
            candidate = np.clip(x + sigma * direction, -5, 5)
            
            try:
                f_candidate = safe_func(candidate)
            except (TimeoutError, StopIteration):
                break
            
            if f_candidate < fx:
                # Success: move to candidate
                x = candidate.copy()
                fx = f_candidate
                success_window.append(1)
                stall_count = 0
            else:
                # Failure: stay
                success_window.append(0)
                stall_count += 1
            
            # Trim window
            if len(success_window) > window_size:
                success_window.pop(0)
            
            # Adapt sigma based on success rate in window
            if len(success_window) >= window_size // 2:
                sr = sum(success_window) / len(success_window)
                # Target success rate ~ 0.2-0.3
                if sr > 0.3:
                    sigma *= 1.15
                elif sr < 0.15:
                    sigma *= 0.85
                # else: keep sigma
            
            sigma = max(sigma_min, min(sigma, sigma_max))
            
            if stall_count >= stall_limit:
                # Aggressive restart: jump significantly from best
                if calls < global_budget and time.time() < deadline and restart_count < max_restarts:
                    # Try a larger jump
                    jump_scale = max(sigma * 3.0, 1.5)
                    x_new = np.clip(x + jump_scale * rng.standard_normal(self.dim), -5, 5)
                    try:
                        fx_new = safe_func(x_new)
                    except (TimeoutError, StopIteration):
                        break
                    if fx_new < fx:
                        x = x_new.copy()
                        fx = fx_new
                    else:
                        # If the jump didn't help, try a smaller one
                        if calls < global_budget and time.time() < deadline:
                            x_new2 = np.clip(x + sigma * 1.5 * rng.standard_normal(self.dim), -5, 5)
                            try:
                                fx_new2 = safe_func(x_new2)
                            except (TimeoutError, StopIteration):
                                break
                            if fx_new2 < fx:
                                x = x_new2.copy()
                                fx = fx_new2
                    sigma = max(sigma * 0.7, sigma_min)
                    stall_count = 0
                    restart_count += 1
        
        # Phase 2: Local Refinement with Nelder-Mead
        # Use remaining budget for local optimization
        remaining = self.budget - calls
        if remaining > 10 and best_x is not None:
            try:
                result = scipy.optimize.minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
            except (TimeoutError, StopIteration, Exception):
                pass
        
        # Phase 3: Adaptive coordinate-descent polish
        remaining = self.budget - calls
        if remaining > 0 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            step = max(sigma, 0.02)
            max_cd_rounds = min(5, remaining // (2 * self.dim))
            
            for _round in range(max_cd_rounds):
                if time.time() > deadline or calls >= self.budget:
                    break
                moved_any = False
                for d in range(self.dim):
                    if calls >= self.budget or time.time() > deadline:
                        break
                    # Try both directions with current step
                    best_trial = None
                    best_trial_f = fx
                    for sign in (1.0, -1.0):
                        if calls >= self.budget:
                            break
                        trial = x.copy()
                        trial[d] = np.clip(trial[d] + sign * step, -5, 5)
                        try:
                            ftrial = safe_func(trial)
                        except (TimeoutError, StopIteration):
                            return best_x, best_f
                        if ftrial < best_trial_f:
                            best_trial_f = ftrial
                            best_trial = trial.copy()
                    if best_trial is not None:
                        x = best_trial
                        fx = best_trial_f
                        moved_any = True
                
                if not moved_any:
                    step *= 0.5
                    if step < 1e-5:
                        break
                # If we moved, keep step; if not, shrink
                if time.time() > deadline:
                    break
        
        return best_x, best_f
