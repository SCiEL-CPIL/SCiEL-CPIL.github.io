# Inside construct, after finding best_z:
    if best_z is not None:
        # Polish with SLSQP
        def objective(z):
            r = z[2::3][:N]
            return -np.sum(r)

        def constraints(z):
            cons = []
            x = z[0::3][:N]
            y = z[1::3][:N]
            r = z[2::3][:N]
            W = z[3*N]
            H = z[3*N + 1]
            
            cons.append({'type': 'ineq', 'fun': lambda: 2.0 - (W + H)})
            
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda i=i: x[i] - r[i]})
                cons.append({'type': 'ineq', 'fun': lambda i=i: W - r[i] - x[i]})
                cons.append({'type': 'ineq', 'fun': lambda i=i: y[i] - r[i]})
                cons.append({'type': 'ineq', 'fun': lambda i=i: H - r[i] - y[i]})
                
            for i in range(N):
                for j in range(i+1, N):
                    cons.append({
                        'type': 'ineq', 
                        'fun': lambda i=i, j=j: (x[i]-x[j])**2 + (y[i]-y[j])**2 - (r[i]+r[j])**2
                    })
            return cons

        try:
            res = minimize(
                objective, 
                best_z, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if res.success:
                x, y, r, W, H = get_vars(res.x)
                if W + H <= 2.0 + 1e-6:
                    # Re-validate
                    valid = True
                    for i in range(N):
                        if x[i] < r[i] - 1e-6 or x[i] > W - r[i] + 1e-6 or y[i] < r[i] - 1e-6 or y[i] > H - r[i] + 1e-6:
                            valid = False; break
                    if valid:
                        for i in range(N):
                            for j in range(i+1, N):
                                if (x[i]-x[j])**2 + (y[i]-y[j])**2 < (r[i]+r[j])**2 - 1e-6:
                                    valid = False; break
                            if not valid: break
                    if valid:
                        val = np.sum(r)
                        if val > best_val:
                            best_val = val
                            best_z = res.x
        except Exception:
            pass
