<<<<<<< SEARCH
def _slsqp_polish(circles, w, h, deadline):
    """
    High-precision polish with SLSQP.
    """
    if time.time() > deadline - 2.0:
        return circles
        
    n = len(circles)
    x0 = np.array([v for c in circles for v in c])
    
    def objective(x):
        return -np.sum(x[2::3])
        
    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-8})
        
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2) - x[i*3+2] - x[j*3+2]
            })
            
    bounds = []
    for i in range(n):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((1e-8, min(w, h)/2))
        
    try:
        res = minimize(
            objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 100, 'ftol': 1e-12}
        )
        if res.success:
            new_circles = np.array([(res.x[i*3], res.x[i*3+1], res.x[i*3+2]) for i in range(n)])
            if _validate(new_circles, w, h, tol=1e-8):
                return new_circles
    except Exception:
        pass
        
    return circles
=======
def _slsqp_polish(circles, w, h, deadline):
    """
    High-precision polish with SLSQP using smooth max for constraints.
    """
    if time.time() > deadline - 2.0:
        return circles
        
    n = len(circles)
    x0 = np.array([v for c in circles for v in c])
    
    # Use smooth max (log-sum-exp) for the overlap constraints to make them differentiable
    # g(x) = max(0, r_i + r_j - dist_ij)
    # We'll use a soft version: log(sum(exp(beta * violations))) / beta for large beta
    # But SLSQP needs inequality constraints. Let's use a different approach:
    # Use the exact constraints but with a tighter tolerance and more iterations.
    # Actually, let's try using a smooth approximation for the distance.
    
    def objective(x):
        return -np.sum(x[2::3])
        
    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i*3] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i*3+1] - x[i*3+2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i*3+2] - 1e-10})
        
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[i*3]-x[j*3])**2 + (x[i*3+1]-x[j*3+1])**2) - x[i*3+2] - x[j*3+2]
            })
            
    bounds = []
    for i in range(n):
        bounds.append((0, w))
        bounds.append((0, h))
        bounds.append((1e-10, min(w, h)/2))
        
    try:
        # First pass with moderate tolerance
        res = minimize(
            objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        if res.success:
            new_circles = np.array([(res.x[i*3], res.x[i*3+1], res.x[i*3+2]) for i in range(n)])
            if _validate(new_circles, w, h, tol=1e-6):
                x0 = res.x
        
        # Second pass with tighter tolerance if time permits
        if time.time() < deadline - 1.0:
            res2 = minimize(
                objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-14}
            )
            if res2.success:
                new_circles = np.array([(res2.x[i*3], res2.x[i*3+1], res2.x[i*3+2]) for i in range(n)])
                if _validate(new_circles, w, h, tol=1e-8):
                    return new_circles
    except Exception:
        pass
        
    return circles
>>>>>>> REPLACE

