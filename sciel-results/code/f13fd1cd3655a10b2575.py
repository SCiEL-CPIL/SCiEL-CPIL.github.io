        calls = 1
        for _ in range(budget - 1):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                successes += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    stall_count = 0
                else:
                    stall_count += 1
            else:
                stall_count += 1
                
            steps_in_window += 1
            
            # Adapt step size
            if steps_in_window >= window:
                success_rate = successes / steps_in_window
                if success_rate > 0.2:
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                successes = 0
                steps_in_window = 0
                
            # Stall handling
            if stall_count > stall_threshold:
                sigma *= 2.0
                stall_count = 0
                
                if rng.random() < 0.1 and calls < budget:
                    x = rng.uniform(-5, 5, dim)
                    fx = func(x)
                    calls += 1
                    sigma = 1.0
                    
        return best_x, best_f
