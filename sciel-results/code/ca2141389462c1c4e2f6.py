            try:
                def soft_obj(params):
                    return soft_score_sq(params.reshape(N, 2), beta)
                
                res = minimize(
                    soft_obj, 
                    current_params.flatten(), 
                    method='L-BFGS-B', 
                    bounds=bnds,
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.fun < 1e9:
                    candidate_pts = res.x.reshape(N, 2)
                    # Convert log-ratio to ratio
                    exact_s = exact_score(candidate_pts)
                    if exact_s < best_score:
                        best_score = exact_s
                        best_params = candidate_pts.copy()
                    current_params = candidate_pts
            except Exception:
                pass
