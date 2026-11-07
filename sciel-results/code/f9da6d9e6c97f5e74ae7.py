        # Strategy: Multi-start
        best_x = None
        best_f = float('inf')
        
        # 1. Start at 0
        x_start = np.zeros(n)
        x_best, f_best = nelder_mead(x_start, 50)
        if f_best < best_f:
            best_f = f_best
            best_x = x_best.copy()
            
        prev_best = best_f
        
        for i in range(1, 10):
            if calls_made >= budget:
                break
            
            # Adaptive Start
            if best_f < prev_best:
                # Improving, stay close
                x_start = best_x + rng.normal(0, 0.5, n)
            else:
                # Stagnating, explore
                x_start = rng.uniform(-5, 5, n)
                
            x_start = np.clip(x_start, -5, 5)
            
            max_evals = calls_made + 50
            if i == 9:
                max_evals = budget
                
            x_best, f_best = nelder_mead(x_start, max_evals)
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()
            prev_best = best_f
