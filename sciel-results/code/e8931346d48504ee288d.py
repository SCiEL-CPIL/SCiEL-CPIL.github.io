        def adaptive_cd(x_start, max_calls):
            nonlocal calls_made, best_x, best_fx
            
            if calls_made >= budget:
                return
                
            x = x_start.copy()
            fx = eval_point(x)
            
            step = 1.0
            min_step = 1e-6
            local_calls = 1 # One call for the start point
            
            while calls_made < budget and local_calls < max_calls and step > min_step:
                improved = False
                for i in range(dim):
                    if calls_made >= budget or local_calls >= max_calls:
                        return
                        
                    # Try positive direction
                    x_new = x.copy()
                    x_new[i] = min(5.0, x_new[i] + step)
                    if not np.allclose(x_new, x):
                        fx_new = eval_point(x_new)
                        local_calls += 1
                        if fx_new < fx:
                            x = x_new
                            fx = fx_new
                            improved = True
                            step *= 0.9
                            continue
                    
                    # Try negative direction
                    x_new = x.copy()
                    x_new[i] = max(-5.0, x_new[i] - step)
                    if not np.allclose(x_new, x):
                        fx_new = eval_point(x_new)
                        local_calls += 1
                        if fx_new < fx:
                            x = x_new
                            fx = fx_new
                            improved = True
                            step *= 0.9
                
                if not improved:
                    step *= 0.5

        # Multi-start points
        starts = [
            np.zeros(dim),
            np.full(dim, 3.0),
            np.full(dim, -3.0),
            np.array([2.0, -2.0, 2.0, -2.0, 2.0][:dim])
        ]

        # Allocate budget
        # Give 40% to first, 30% to second, 20% to third, 10% to fourth.
        # Or simply split evenly.
        budget_per_start = budget // len(starts)
        
        for i, start_x in enumerate(starts):
            if calls_made >= budget:
                break
            
            if i == len(starts) - 1:
                max_calls = budget - calls_made
            else:
                max_calls = budget_per_start
            
            adaptive_cd(start_x, max_calls)

        # Final refinement from best_x found
        if best_x is not None and calls_made < budget:
            remaining = budget - calls_made
            adaptive_cd(best_x, remaining)

        return best_x, best_fx
