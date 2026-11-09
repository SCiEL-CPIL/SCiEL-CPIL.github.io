        def run_cmaes(x0, budget):
            x = x0.copy()
            fx = func(x)
            best_x = x.copy()
            best_f = fx
            sigma = 1.0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            
            calls = 1
            
            for _ in range(budget - 1):
                if time.time() > deadline - 10:
                    break
                
                # ... generation and evaluation ...
                
                # Update best
                if f_parents[0] < best_f:
                    best_f = f_parents[0]
                    best_x = y_parents[0].copy()
                
                # ... updates ...
                
            return best_x, best_f
