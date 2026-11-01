<<<<<<< SEARCH
    # Refine if time permits
    if time.time() < deadline - 30:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = np.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = np.radians(best_result[3])
            
            # Final refinement with exact objective
            res = minimize(lambda p: objective(p, use_smooth=False), params_best, 
                          method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < best_score:
                params = res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append((cx, cy, ang))
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = np.degrees(params[3*N+2]) % 360
                S = compute_required_side(inner, (ox, oy), outer_angle)
                
                if check_validity(inner, (ox, oy), S, outer_angle):
                    best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass
=======
    # Refine if time permits
    if time.time() < deadline - 30:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = np.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = np.radians(best_result[3])
            
            # Final refinement with exact objective
            res = minimize(lambda p: objective(p, use_smooth=False), params_best, 
                          method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            
            params = res.x
            inner = []
            for i in range(N):
                cx = params[3*i]
                cy = params[3*i+1]
                ang = np.degrees(params[3*i+2]) % 360
                inner.append((cx, cy, ang))
            ox = params[3*N]
            oy = params[3*N+1]
            outer_angle = np.degrees(params[3*N+2]) % 360
            S = compute_required_side(inner, (ox, oy), outer_angle)
            
            if check_validity(inner, (ox, oy), S, outer_angle, eps=1e-5):
                if S < best_score:
                    best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass
>>>>>>> REPLACE

