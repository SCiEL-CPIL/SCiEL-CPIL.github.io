def _optimize_single(W, H, x0_flat, maxiter=200):
    bounds = [(0, W)] * N + [(0, H)] * N + [(0, min(W, H) / 2.0)] * N
    
    cons = ({
        'type': 'ineq',
        'fun': lambda x: _combined_constraints(x, W, H),
        'jac': lambda x: _combined_jacobian(x, W, H)
    },)
    
    try:
        res = minimize(
            _obj_flat, x0_flat, method='SLSQP', 
            jac=_obj_grad_flat, 
            bounds=bounds, 
            constraints=cons,
            options={'maxiter': maxiter, 'ftol': 1e-9}
        )
        
        x_opt = res.x
        cx = x_opt[0::3]
        cy = x_opt[1::3]
        r = x_opt[2::3]
        
        # Quick validity check
        valid = True
        for i in range(N):
            if r[i] < -1e-5: valid = False; break
            if cx[i] < r[i] - 1e-4 or cx[i] > W - r[i] + 1e-4: valid = False; break
            if cy[i] < r[i] - 1e-4 or cy[i] > H - r[i] + 1e-4: valid = False; break
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    d2 = (cx[i]-cx[j])**2 + (cy[i]-cy[j])**2
                    if d2 < (r[i]+r[j])**2 - 1e-4:
                        valid = False
                        break
                if not valid: break
        
        if valid:
            # Clip tiny negative radii to 0
            r = np.maximum(r, 0)
            x_opt[2::3] = r
            return x_opt, -res.fun
        else:
            return None, 0.0
    except Exception:
        return None, 0.0
