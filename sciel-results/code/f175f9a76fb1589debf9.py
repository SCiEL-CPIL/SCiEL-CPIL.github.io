<<<<<<< SEARCH
            # Multi-stage: start with high beta (smoother), then decrease
            betas = [50.0, 20.0, 10.0, 5.0, 0.0]
            current_params = init_params.copy()
            
            for beta in betas:
                if time.time() - t0 > DEADLINE - 3.0:
                    break
                try:
                    if beta > 0:
                        res = minimize(lambda p: polish_objective_smooth(p, beta), current_params, 
                                       method='Nelder-Mead', 
                                       options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7})
                    else:
                        res = minimize(lambda p: polish_objective_smooth(p, 0.0), current_params, 
                                       method='Nelder-Mead', 
                                       options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8})
                    
                    if res.fun < 1e5:  # Valid result
                        current_params = res.x.copy()
                except Exception:
                    pass
>>>>>>> REPLACE

