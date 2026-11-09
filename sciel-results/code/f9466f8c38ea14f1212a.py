import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Time management with safety margin
        start_time = time.time()
        time_limit = 320.0
        
        # Global best tracking
        x_best = np.zeros(n)
        f_best = np.inf
        eval_count = 0
        
        def try_eval(x):
            """Evaluate x, update global best, return fx."""
            nonlocal f_best, x_best, eval_count
            fx = func(x)
            eval_count += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx
        
        # Phase 1: Initialization - evaluate center + 9 random points
        init_points = [np.zeros(n)]
        for _ in range(9):
            init_points.append(np.random.uniform(-5, 5, n))
        
        for x_init in init_points:
            if time.time() - start_time > time_limit or eval_count >= budget:
                break
            try:
                try_eval(x_init)
            except Exception:
                break
        
        # Phase 2: Main search - (1+1)-ES with coordinate descent refinement
        sigma_min = 1e-10
        sigma_max = 8.0
        restart_threshold = 30  # Steps without global improvement before restart
        
        while eval_count < budget:
            if time.time() - start_time > time_limit:
                break
            
            # Start this restart from the global best
            x = x_best.copy()
            fx = f_best
            sigma = 1.5
            steps_since_best = 0
            
            # (1+1)-ES local search
            while eval_count < budget:
                if time.time() - start_time > time_limit:
                    break
                
                # Generate candidate
                z = np.random.standard_normal(n)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                
                try:
                    fy = func(y)
                except Exception:
                    break
                eval_count += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2
                    if fy < f_best:
                        f_best = fy
                        x_best = x.copy()
                        steps_since_best = 0
                    else:
                        steps_since_best += 1
                else:
                    sigma *= 0.85
                    steps_since_best += 1
                
                # Clamp sigma
                sigma = np.clip(sigma, sigma_min, sigma_max)
                
                # Check for stagnation
                if steps_since_best >= restart_threshold:
                    break
                
                # Early termination if sigma is very small and no recent improvement
                if sigma < 0.02 and steps_since_best > 15:
                    break
            
            # Coordinate-descent refinement on the current best
            if eval_count < budget and time.time() - start_time < time_limit:
                x_cd = x_best.copy()
                fx_cd = f_best
                max_cd_passes = 3
                
                for pass_num in range(max_cd_passes):
                    if eval_count >= budget or time.time() - start_time > time_limit:
                        break
                    
                    improved = False
                    for dim in range(n):
                        if eval_count >= budget or time.time() - start_time > time_limit:
                            break
                        
                        # Binary search along this dimension
                        lo = -5.0
                        hi = 5.0
                        x_test = x_cd.copy()
                        
                        # First, try to find a better value by scanning
                        # Use a few probe points to bracket the minimum
                        best_val = fx_cd
                        best_pos = x_cd[dim]
                        
                        # Probe at current value and a few offsets
                        probes = [best_pos - 0.5, best_pos - 0.1, best_pos, best_pos + 0.1, best_pos + 0.5]
                        for p in probes:
                            if p < lo or p > hi:
                                continue
                            x_test[dim] = p
                            try:
                                fv = func(x_test)
                                eval_count += 1
                                if fv < best_val:
                                    best_val = fv
                                    best_pos = p
                                    improved = True
                            except Exception:
                                pass
                        
                        # If we found improvement, do a finer binary search around best_pos
                        if improved and best_pos != x_cd[dim]:
                            # Narrow the range around best_pos
                            center = best_pos
                            width = 0.5
                            for _ in range(4):  # 4 iterations of binary search
                                if eval_count >= budget or time.time() - start_time > time_limit:
                                    break
                                
                                mid = center
                                # Try left and right
                                left = center - width
                                right = center + width
                                
                                left_ok = left >= lo
                                right_ok = right <= hi
                                
                                if left_ok and right_ok:
                                    x_test[dim] = left
                                    try:
                                        fv_left = func(x_test)
                                        eval_count += 1
                                    except Exception:
                                        fv_left = np.inf
                                    
                                    x_test[dim] = right
                                    try:
                                        fv_right = func(x_test)
                                        eval_count += 1
                                    except Exception:
                                        fv_right = np.inf
                                    
                                    if fv_left < best_val:
                                        best_val = fv_left
                                        center = left
                                        improved = True
                                    elif fv_right < best_val:
                                        best_val = fv_right
                                        center = right
                                        improved = True
                                    else:
                                        # Keep center
                                        pass
                                elif left_ok:
                                    x_test[dim] = left
                                    try:
                                        fv = func(x_test)
                                        eval_count += 1
                                        if fv < best_val:
                                            best_val = fv
                                            center = left
                                            improved = True
                                    except Exception:
                                        pass
                                elif right_ok:
                                    x_test[dim] = right
                                    try:
                                        fv = func(x_test)
                                        eval_count += 1
                                        if fv < best_val:
                                            best_val = fv
                                            center = right
                                            improved = True
                                    except Exception:
                                        pass
                                
                                width *= 0.5
                            
                            x_cd[dim] = center
                            fx_cd = best_val
                            if fx_cd < f_best:
                                f_best = fx_cd
                                x_best = x_cd.copy()
                        
                        if not improved:
                            break  # No improvement in this dimension, stop CD pass
        
        return x_best, f_best
